# -*- coding: utf-8 -*-
"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression/classification using JAX.

(C) 2024-2026 A. Bemporad
"""

import numpy as np
import jax
import jax.numpy as jnp
from joblib import Parallel, delayed, cpu_count
from ._model import Model


class LinearModel(Model):
    def __init__(self, nx, ny, nu, feedthrough=False, y_in_x=False, x0=None, sigma=0.5, seed=0, Ts=None, ss=None, stability=False):
        """Create the linear state-space model structure

            x(k+1) = A*x(k) + B*u(k)
              y(k) = C*x(k) + D*u(k)

        Parameters
        ----------
        nx : int
            Number of states.
        ny : int
            Number of outputs.
        nu : int
            Number of inputs.
        feedthrough : bool
            If True, the output depends also on the input u(k), i.e., D is not zero.
        y_in_x : bool
            If True, the output is included in the first ny components of the state vector
        x0 : ndarray or list of ndarrays or None
            Initial state vector or list of initial state vectors for multiple experiments. If None, x0=0.
        sigma : float
            Initial A matrix = sigma*I, where I is the identity matrix
        seed : int
            Random seed for initialization (default: 0)
        Ts : float
            Sampling time
        ss : list of ndarray or None
            Initial value of A, B, C, D matrices for state-space realization (optional)
            If y_in_x=True, provided values for C and D are ignored.
            If y_in_x=False and feedthrough=False, provided value for D is ignored.
        stability: bool
            If True, set the state-transition matrix to A/max(abs(eig(A)),1) to guarantee system stability.
        """

        super().__init__(nx, ny, nu, state_fcn=None, output_fcn=None, y_in_x=y_in_x, Ts=Ts)

        self.isLinear = True
        self.feedthrough = feedthrough
        if ss is not None:
            if y_in_x:
                params = ss[0:2]
            else:
                if feedthrough:
                    params = ss
                else:
                    params = ss[0:3]
        else:
            params = None
        self.init(params=params, x0=x0, sigma=sigma, seed=seed)
        self.sigma = sigma  # required by parallel_fit()

        self.isStable = stability # If True, A matrix is trained as A/max(|eig(A)|,1) to enforce stability
        if not self.isStable:
            @jax.jit
            def state_fcn(x, u, params):
                A, B = params[0:2]
                return A @ x + B @ u
            @jax.jit
            def stabilize(A):
                return A
        else:
            # @jax.jit
            # def stabilize(A):
            #     eigA = jnp.linalg.eigvals(A)
            #     maxeig = jnp.max(jnp.abs(eigA)) # This could be made more efficient by computing only the dominant eigenvalue
            #     return A / jnp.maximum(maxeig, 1.0)
            @jax.jit
            def stabilize(A):
                return A / jnp.maximum(jnp.linalg.norm(A,2),1.) # More efficient method using matrix 2-norm
            @jax.jit
            def state_fcn(x, u, params):
                A, B = params[0:2]
                A = self.stabilize_fcn(A)
                return A @ x + B @ u
        self.stabilize_fcn = stabilize

        if self.y_in_x:
            @jax.jit
            def output_fcn(x, u, params):
                return x[0:self.ny]
        else:
            if not feedthrough:
                def output_fcn(x, u, params):
                    C = params[2]
                    return C @ x
            else:
                @jax.jit
                def output_fcn(x, u, params):
                    C, D = params[2:4]
                    return C @ x + D @ u

        self.state_fcn = state_fcn
        self.output_fcn = output_fcn

        return

    def params2ABCD(self):
        A, B = self.params[0:2]
        if self.isStable:
            A = self.stabilize_fcn(A)
        if self.y_in_x:
            C = np.hstack(
                (np.eye(self.ny), np.zeros((self.ny, self.nx-self.ny))))
            D = np.zeros((self.ny, self.nu))
        else:
            C = self.params[2]
            if self.feedthrough:
                D = self.params[3]
            else:
                D = np.zeros((self.ny, self.nu))
        return A, B, C, D

    def ssdata(self):
        """Retrieve state-space realization of the linear system

            x(k+1) = A*x(k) + B*u(k)
            y(k) = C*x(k) + D*u(k)

        Returns
        -------
        A : ndarray
            A matrix
        B : ndarray
            B matrix
        C : ndarray
            C matrix
        D : ndarray
            D matrix
        """
        # Retrieve state-space realization
        A, B, C, D = self.params2ABCD()
        return A, B, C, D

    def group_lasso_x(self):
        """Group-Lasso regularization on linear state-space matrices A,B,C and x0 to penalize the model order (number of states)
        """
        @jax.jit
        def groupLassoRegX(th, x0):
            cost = 0.
            A, B = th[0:2]
            if self.isStable:
                A = self.stabilize_fcn(A)
            if self.y_in_x:
                for i in range(self.nx):
                    cost += jnp.sqrt(jnp.sum(A[:, i]**2)+jnp.sum(A[i, :]**2) -
                                     A[i, i]**2+jnp.sum(B[i, :]**2) + sum([x0i[i]**2 for x0i in x0]))
            else:
                C = th[2]
                for i in range(self.nx):
                    cost += jnp.sqrt(jnp.sum(A[:, i]**2)+jnp.sum(A[i, :]**2)-A[i, i]**2+jnp.sum(
                        B[i, :]**2) + jnp.sum(C[:, i]**2)+sum([x0i[i]**2 for x0i in x0]))
            return cost
        self.group_lasso_fcn = groupLassoRegX
        return

    def group_lasso_u(self):
        """Group-Lasso regularization on linear state-space matrices B,D to penalize the number of inputs entering the model.
        """
        @jax.jit
        def groupLassoRegU(th, x0):  # x0 is ignored, just for interface compatibility
            cost = 0.
            B = th[1]
            if not self.y_in_x and self.feedthrough:
                D = th[3]
                for i in range(self.nu):
                    cost += jnp.sqrt(jnp.sum(B[:, i]
                                             ** 2)+jnp.sum(D[:, i]**2))
            else:
                for i in range(self.nu):
                    cost += jnp.sqrt(jnp.sum(B[:, i]**2))
            return cost
        self.group_lasso_fcn = groupLassoRegU
        return

    def force_stability(self, rho_A=1.e3, epsilon_A=1.e-3):
        """Force stability of the linear state-space model by imposing the soft constraint ||A||_2 <= 1 as the state-transition matrix.

        The constraint ||A||_2 <= 1 is mapped into the following custom regularization term in the optimization problem:

        rho_A * max{||A||_2^2 − 1 + epsilon_A, 0}^2

        where rho_A is a large penalty and epsilon_A is a small positive number used to tighten the constraint.

        Parameters
        ----------
        rho_A : float
            Penalty coefficient
        epsilon_A : float
            Tolerance for the constraint ||A||_2 <= 1
        """

        if self.isStable:
            print("\033[1mWarning: the model was already forced to be stable. Changing stability enforcement method.\033[0m")
            @jax.jit
            def stabilize(A):
                return A
            self.stabilize_fcn = stabilize
            self.isStable = False # Disable existing stability enforcement, now relies on penalty function

        @jax.jit
        def force_stability(th, x0):
            A = th[0]
            return rho_A*jnp.maximum(jnp.linalg.norm(A, 2)**2-1.+epsilon_A, 0.)**2

        self.custom_regularization = force_stability

    def parallel_fit(self, Y, U, seeds, n_jobs=None):
        """
        Fits the model in parallel using multiple seeds.

        Parameters:
            Y : ndarray or list of ndarrays
                Training dataset: output data. Y must be a N-by-ny numpy array
                or a list of Ni-by-ny numpy arrays, where Ni is the length of the i-th experiment.
            U : ndarray
                Training dataset: input data. U must be a N-by-nu numpy array
                or a list of Ni-by-nu numpy arrays, where Ni is the length of the i-th experiment.
            seeds (array-like): The seeds used for initialization.
            n_jobs (int): The number of parallel jobs to run (default is None, which means using all available cores).

        Returns:
            list: A list of fitted models.
        """
        def single_fit(seed):
            if not jax.config.jax_enable_x64:
                # Enable 64-bit computations
                jax.config.update("jax_enable_x64", True)
            self.init(sigma=self.sigma, seed=seed)
            # if self.iprint > -1:
            #     print(
            #         "\033[1m" + f"Fitting model with seed = {seed} ... " + "\033[0m")
            self.fit(Y, U)
            # if self.iprint > -1:
            #     print("\033[1m" + f"Seed = {seed}: done." + "\033[0m")
            return self

        if n_jobs is None:
            n_jobs = cpu_count()

        return Parallel(n_jobs=n_jobs)(delayed(single_fit)(seed) for seed in seeds)

    def dcgain_loss(self, Uss=None, Yss=None, beta=1., DCgain=None):
        """
        Define a loss function for fitting the DC gain of a linear model to either a given matrix or to given steady-state input-output pairs.

        Parameters:
        -----------
        Uss : array-like or None
            Steady-state input values. If None, the DC gain loss must be computed from the desired DC gain matrix DCgain.
        Yss : array-like or None
            Steady-state output values.
        beta : float, optional
            Penalty on the DC gain loss
        DCgain : array-like or None
            Desired DC gain matrix. If None, the DC gain loss must be computed from the steady-state input-output pairs Uss, Yss.

        Returns:
        --------
        dcgain_loss : function
            A function that computes the DC gain loss given model parameters. This function can be used a custom regularization loss in the fit method.
        """

        if DCgain is not None:
            if DCgain.shape[0] != self.ny or DCgain.shape[1] != self.nu:
                raise ValueError(
                    "DCgain matrix has wrong dimensions. Expected %d-by-%d" % (self.ny, self.nu))

            def dcgain_loss(params, x0):
                A, B = params[0:2]
                A = self.stabilize_fcn(A)
                ssgain = jnp.linalg.solve(jnp.eye(self.nx)-A, B)
                if self.y_in_x:
                    dcgain = ssgain[0:self.ny]
                else:
                    dcgain = params[2]@ssgain
                    if self.feedthrough:
                        dcgain += params[3]
                return beta*jnp.sum((dcgain-DCgain)**2)
        else:
            if Uss is None or Yss is None:
                raise ValueError(
                    "Steady-state data Uss and Yss must be provided if DCgain is not given")
            dcgain_loss = super().dcgain_loss(Uss, Yss, beta)
        return dcgain_loss
