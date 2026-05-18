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
from ._linear_model import LinearModel


class qLPVModel(Model):
    def __init__(self, nx, ny, nu, npar, qlpv_fcn, qlpv_params_init, feedthrough=False, y_in_x=False, x0=None, sigma=0.5, seed=0, Ts=None):
        """Create the quasi-LPV (Linear Parameter Varying) state-space model structure

            x(k+1) = A(p(k))*x(k) + B(p(k))*u(k)
              y(k) = C(p(k))*x(k) + D(p(k))*u(k)

              p(k) = f(x(k),u(k),qlpv_params),   dim(p(k)) = npar

              A(p(k)) = A_lin + ∑_{i=1}^{n_p} A_i p(k)_i,   B(p(k)) = B_lin + ∑_{i=1}^{n_p} B_i p(k)_i
              C(p(k)) = C_lin + ∑_{i=1}^{n_p} C_i p(k)_i,   D(p(k)) = D_lin + ∑_{i=1}^{n_p} D_i p(k)_i

        Parameters
        ----------
        nx : int
            Number of states
        ny : int
            Number of outputs
        nu : int
            Number of inputs
        npar : int
            Number of parameters defining the scheduling function f(x,u,theta)
        qlpv_fcn : callable
            Function defining the scheduling function f(x,u,qlpv_params)
        qlpv_params_init : list of ndarrays
            Initial values of the arrays defining the scheduling function parameters qlpv_params
        feedthrough : bool
            If True, the output depends also on the input u(k), i.e., D(p(k)) is not zero.
        y_in_x : bool
            If True, the output is included in the first ny components of the state vector
        x0 : ndarray or list of ndarrays or None
            Initial state vector or list of initial state vectors for multiple experiments. If None, x0=0.
        sigma : float
            Initial Alin matrix = sigma*I, where I is the identity matrix. Each other matrix in Ap is initialized
            as 0.2/(1+npar)*I.
        seed : int
            Random seed for initialization (default: 0)
        Ts : float
            Sampling time
        """

        super().__init__(nx, ny, nu, state_fcn=None, output_fcn=None, y_in_x=y_in_x, Ts=Ts)

        self.isqLPV = True
        self.feedthrough = feedthrough
        self.nqLPVpar = npar
        nqlpv_params = len(qlpv_params_init)
        self.nqlpv_params = nqlpv_params

        self.init(qlpv_params_init, sigma, seed, x0)
        self.isInitialized = True
        self.sigma = sigma  # required by parallel_fit()

        @jax.jit
        def state_fcn(x, u, params):
            qlpv_params = params[:nqlpv_params]
            p = qlpv_fcn(x, u, qlpv_params)
            Alin = params[nqlpv_params]
            Ap = params[nqlpv_params+1]
            Blin = params[nqlpv_params+2]
            Bp = params[nqlpv_params+3]
            x = Alin@x + Blin@u + p@(Ap@x + Bp@u)
            return x

        if self.y_in_x:
            @jax.jit
            def output_fcn(x, u, params):
                return x[0:self.ny]
        else:
            @jax.jit
            def output_fcn(x, u, params):
                qlpv_params = params[:nqlpv_params]
                p = qlpv_fcn(x, u, qlpv_params)
                Clin = params[nqlpv_params+4]
                Cp = params[nqlpv_params+5]
                y = Clin@x + p@(Cp@x)
                if feedthrough:
                    Dlin = params[nqlpv_params+6]
                    Dp = params[nqlpv_params+7]
                    y += Dlin@u + p@(Dp@u)
                return y

        self.state_fcn = state_fcn
        self.output_fcn = output_fcn
        return

    def init(self, qlpv_params_init, sigma, seed, x0=None):

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)

        if x0 is not None:
            if isinstance(x0, list):
                Nexp = len(x0)
                self.x0 = [jnp.array(x0[i]) for i in range(Nexp)]
            else:
                self.x0 = jnp.array(x0)
        else:
            self.x0 = jnp.zeros(self.nx)

        params = qlpv_params_init.copy()

        lin_model = LinearModel(
            self.nx, self.ny, self.nu, feedthrough=self.feedthrough, y_in_x=self.y_in_x, x0=self.x0, Ts=self.Ts)
        lin_model.init(sigma=sigma, seed=seed)
        Alin, Blin, Clin, Dlin = lin_model.params2ABCD()

        npar = self.nqLPVpar
        Ap = 0.2/(1.+npar)*jnp.kron(jnp.ones((npar, 1, 1)), jnp.eye(self.nx))
        # this avoids using the same seed used for LTI model
        key1, key2 = jax.random.split(jax.random.PRNGKey(seed+1), 2)
        Bp = 0.1 * jax.random.normal(key1, (npar, self.nx, self.nu))
        params.extend([Alin, Ap, Blin, Bp])

        if not self.y_in_x:
            Cp = 0.1 * jax.random.normal(key2, (npar, self.ny, self.nx))
            params.extend([Clin, Cp])
            if self.feedthrough:
                Dp = jnp.zeros((npar, self.ny, self.nu))
                params.extend([Dlin, Dp])
        self.params = params
        return

    def ssdata(self):
        """Retrieve state-space realization of the quasi-LPV system

            x(k+1) = A(p(k))*x(k) + B(p(k))*u(k)
              y(k) = C(p(k))*x(k) + D(p(k))*u(k)

              p(k) = f(x(k),u(k),qlpv_params),   dim(p(k)) = npar

              A(p(k)) = A_lin + ∑_{i=1}^{n_p} A_i p(k)_i,   B(p(k)) = B_lin + ∑_{i=1}^{n_p} B_i p(k)_i
              C(p(k)) = C_lin + ∑_{i=1}^{n_p} C_i p(k)_i,   D(p(k)) = D_lin + ∑_{i=1}^{n_p} D_i p(k)_i

        Returns
        -------
        Alin : ndarray
            baseline A matrix
        Blin : ndarray
            baseline B matrix
        Clin : ndarray
            baseline C matrix
        Dlin : ndarray
            baseline D matrix
        Ap : ndarray
            parameter-dependent A matrix components
        Bp : ndarray
            parameter-dependent B matrix components
        Cp : ndarray
            parameter-dependent C matrix components
        Dp : ndarray
            parameter-dependent D matrix components
        """
        nqlpv_params = self.nqlpv_params
        Alin = self.params[nqlpv_params]
        Ap = self.params[nqlpv_params+1]
        Blin = self.params[nqlpv_params+2]
        Bp = self.params[nqlpv_params+3]

        Clin = np.hstack(
            (np.eye(self.ny), np.zeros((self.ny, self.nx-self.ny))))
        Cp = np.zeros((self.nqLPVpar, self.ny, self.nx))
        Dlin = np.zeros((self.nqLPVpar, self.ny, self.nu))
        Dp = np.zeros((self.nqLPVpar, self.ny, self.nu))
        if not self.y_in_x:
            Clin = self.params[nqlpv_params+4]
            Cp = self.params[nqlpv_params+5]
            if self.feedthrough:
                Dlin = self.params[nqlpv_params+6]
                Dp = self.params[nqlpv_params+7]

        return Alin, Blin, Clin, Dlin, Ap, Bp, Cp, Dp

    def fit(self, Y, U, LTI_training=True):
        """
        Train a dynamical qLPV model using input-output data.

        Parameters:
        ----------
        Y : ndarray or list of ndarrays
            Training dataset: output data. Y must be a N-by-ny numpy array
            or a list of Ni-by-ny numpy arrays, where Ni is the length of the i-th experiment.
        U : ndarray
            Training dataset: input data. U must be a N-by-nu numpy array
            or a list of Ni-by-nu numpy arrays, where Ni is the length of the i-th experiment.
        LTI_training : bool
            If True, the LTI matrices Alin, Blin, Clin, Dlin are trained first and used as an initial guess.
        """

        nx = self.nx
        if nx < 1:
            raise (
                Exception("\033[1mModel order 'nx' must be greater than zero\033[0m"))

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)

        if LTI_training:
            Alin, Blin, Clin, Dlin, _, _, _, _ = self.ssdata()
            lin_model = LinearModel(nx, self.ny, self.nu, feedthrough=self.feedthrough,
                                    y_in_x=self.y_in_x, x0=self.x0, Ts=self.Ts, ss=[Alin, Blin, Clin, Dlin])
            if self.rho_x0 is not None:
                rho_x0 = self.rho_x0
            else:
                rho_x0 = 1.e-4
            if self.rho_th is not None:
                rho_th = self.rho_th
            else:
                rho_th = 1.e-4
            lin_model.loss(rho_x0=rho_x0, rho_th=rho_th)
            if self.iprint is not None:
                iprint = self.iprint
            else:
                iprint = 50
            if iprint > 0:
                print("\n\nTraining LTI model...\n\n")
            lin_model.optimization(
                adam_epochs=0, lbfgs_epochs=2000, iprint=iprint)
            lin_model.fit(Y, U)
            Alin, Blin, Clin, Dlin = lin_model.params2ABCD()

            nqlpv_params = self.nqlpv_params
            self.params[nqlpv_params] = Alin
            self.params[nqlpv_params+2] = Blin
            if not self.y_in_x:
                self.params[nqlpv_params+4] = Clin
                if self.feedthrough:
                    self.params[nqlpv_params+6] = Dlin

            super().fit(Y, U)
        return

    def group_lasso_p(self):
        """Group-Lasso regularization on quasi-LPV model matrices Ap, Bp, Cp, Dp to penalize the number of scheduling parameters.
        """

        @jax.jit
        def groupLassoRegP(th, x0):

            nqlpv_params = self.nqlpv_params
            Ap = th[nqlpv_params+1]
            Bp = th[nqlpv_params+3]
            if not self.y_in_x:
                Cp = th[nqlpv_params+5]
                if self.feedthrough:
                    Dp = th[nqlpv_params+7]

            cost = 0.
            for i in range(self.nqLPVpar):
                cost_i = jnp.sum(Ap[i]**2)+jnp.sum(Bp[i]**2)
                if not self.y_in_x:
                    cost_i += jnp.sum(Cp[i]**2)
                    if self.feedthrough:
                        cost_i += jnp.sum(Dp[i]**2)
                cost += jnp.sqrt(cost_i)
            return cost

        self.group_lasso_fcn = groupLassoRegP
        return

    def parallel_fit(self, Y, U, qlpv_param_init_fcn, seeds, n_jobs=None, LTI_training=True):
        """
        Fits the model in parallel using multiple seeds.

        Parameters:
            Y : ndarray or list of ndarrays
                Training dataset: output data. Y must be a N-by-ny numpy array
                or a list of Ni-by-ny numpy arrays, where Ni is the length of the i-th experiment.
            U : ndarray
                Training dataset: input data. U must be a N-by-nu numpy array
                or a list of Ni-by-nu numpy arrays, where Ni is the length of the i-th experiment.
            qlpv_param_init_fcn : callable
                Function to initialize the parameters of the scheduling function qlpv_params given a seed.
            seeds (array-like): The seeds used for initialization.
            n_jobs (int): The number of parallel jobs to run (default is None, which means using all available cores).
            LTI_training : bool
            If True, the LTI matrices Alin, Blin, Clin, Dlin are trained first and used as an initial guess.

        Returns:
            list: A list of fitted models.
        """
        def single_fit(seed):
            if not jax.config.jax_enable_x64:
                # Enable 64-bit computations
                jax.config.update("jax_enable_x64", True)
            qlpv_params_init = qlpv_param_init_fcn(seed)
            self.init(qlpv_params_init, self.sigma, seed, x0=None)
            # if self.iprint > -1:
            #     print(
            #         "\033[1m" + f"Fitting model with seed = {seed} ... " + "\033[0m")
            self.fit(Y, U, LTI_training=LTI_training)
            # if self.iprint > -1:
            #     print("\033[1m" + f"Seed = {seed}: done." + "\033[0m")
            return self

        if n_jobs is None:
            n_jobs = cpu_count()  # Use all available cores by default

        models = Parallel(n_jobs=n_jobs)(
            delayed(single_fit)(seed=seed) for seed in seeds)
        return models
