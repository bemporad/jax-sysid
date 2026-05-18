# -*- coding: utf-8 -*-
"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression/classification using JAX.

(C) 2024-2026 A. Bemporad
"""

import numpy as np
import time
import jax
import jax.numpy as jnp
import jaxopt
from functools import partial
from jax_sysid.utils import lbfgs_options, vec_reshape, compute_scores
from joblib import Parallel, delayed, cpu_count
from ._solvers import (
    EPSIL_LASSO, DEFAULT_SMALL_TAU_TH,
    l2reg, l1reg, linreg,
    optimization_base, adam_solver, lbfgs_callback,
    xsat, get_bounds,
)


class Model(object):
    """
    Base class of dynamical models for system identification
    """

    def __init__(self, nx, ny, nu, state_fcn=None, output_fcn=None, y_in_x=False, Ts=None):
        """
        Initialize model structure.

        Parameters
        ----------
        nx : int
            Number of states.
        ny : int
            Number of outputs.
        nu : int
            Number of inputs.
        state_fcn : callable, optional
            Function handle to the state-update function x(k+1)=state_fcn(x(k),u(k),params).
            If not None, an output function "output_fcn" must be also provided.
            Use state_fcn=None and output_fcn=None to get a linear model.
        output_fcn : callable, optional
            Function handle to the output function y(k)=output_fcn(x(k),u(k),params).
            If not None, a state-update function "state_fcn" must be also provided.
            Use state_fcn=None and output_fcn=None to get a linear model.
        y_in_x : bool, optional
            If True, the output is equal to the first ny states, i.e., y = [I 0]x.
            This forces nx>=ny. The output function output_fcn is ignored.
        Ts : float, optional
            Sample time (default: 1 time unit).
        """
        self.nx = nx  # number of states
        self.ny = ny  # number of outputs
        self.nu = nu  # number of inputs
        self.y_in_x = y_in_x
        self.state_fcn = state_fcn  # state function
        self.output_fcn = output_fcn

        if y_in_x:
            self.nx = max(nx, ny)  # force nx>=ny

            def output_fcn(x, u, params):
                return x[0:ny]
            self.output_fcn = output_fcn
        self.isLinear = False  # By default, the model is a general nonlinear model
        self.isqLPV = False
        self.Ts = Ts  # sample time

        self.loss()  # define default loss function
        self.optimization()  # define default optimization parameters

        self.isInitialized = False  # model parameters have not been initialized yet

        self.x0 = None
        self.Jopt = None
        self.t_solve = None
        self.sat_activated = None
        self.sparsity = None
        self.group_lasso_fcn = None
        self.custom_regularization = None

        self.params_min = None
        self.params_max = None
        self.x0_min = None
        self.x0_max = None
        self.isbounded = None

    def predict(self, x0, U, qx=0., qy=0.):
        """
        Simulate the response of a dynamical system.

        Parameters
        ----------
        x0 : ndarray
            Initial state vector.
        U : ndarray
            Input signal. U must be a N-by-nu numpy array.
        qx : float
            Standard deviation of process noise (default: 0.0).
        qy : float
            Standard deviation of measurement noise (default: 0.0).

        Returns
        -------
        Y : ndarray
            Output signal (N-by-ny numpy array).
        X : ndarray
            State trajectory (N-by-nx numpy array).
        """
        U = vec_reshape(U)
        x = x0.copy().reshape(-1)
        nx = self.nx
        N = U.shape[0]
        ny = self.ny

        use_scan = (qx is None or qx == 0.) and (qy is None or qy == 0.)

        if not use_scan:
            # simulate with noise in a for-loop
            Y = np.empty((N, ny))
            X = np.empty((N, nx))
            for k in range(N):
                u = U[k]
                Y[k] = self.output_fcn(x, u, self.params) + \
                    qy * np.random.randn(ny)
                X[k] = x
                x = self.state_fcn(x, u, self.params) + \
                    qx * np.random.randn(nx)
        else:
            @jax.jit
            def model_step(x, u):
                y = jnp.hstack((self.output_fcn(x, u, self.params), x))
                x = self.state_fcn(x, u, self.params).reshape(-1)
                return x, y
            _, YX = jax.lax.scan(model_step, x, U)
            Y = YX[:, 0:ny]
            X = YX[:, ny:]

        return Y, X

    def loss(self, output_loss=None, rho_x0=0.001, rho_th=0.01, tau_th=0.0, tau_g=0.0, group_lasso_fcn=None, zero_coeff=0., xsat=1000., train_x0=True, custom_regularization=None):
        """Define the overall loss function for system identification

        Parameters
        ----------
        output_loss : function
            Loss function penalizing output fit errors, loss=output_loss(Yhat,Y), where Yhat is the sequence of predicted outputs and Y is the measured output.
            If None, use standard mean squared error loss=sum((Yhat-Y)**2)/Y.shape[0]
        rho_x0 : float
            L2-regularization on initial state
        rho_th : float
            L2-regularization on model parameters
        tau_th : float
            L1-regularization on model parameters
        tau_g : float
            group-Lasso regularization penalty
        group_lasso_fcn : function
            function f(params,x0) defining the group-Lasso penalty on the model parameters "params" and initial state "x0", minimized as tau_g*sum(||[params;x0]_i||_2). For linear models, you can use the "group_lasso_x" or "group_lasso_u" methods.
        zero_coeff : _type_
            Entries smaller than zero_coeff are set to zero. Useful when tau_th>0 or tau_g>0.
        xsat : float
            Saturation value for state variables, forced during training to avoid numerical issues.
        train_x0 : bool
            If True, also train the initial state x0, otherwise set x0=0 and ignore rho_x0.
        custom_regularization : function
            Additional custom regularization term, a function of the model parameters and initial state
            custom_regularization(params, x0).
        """
        if output_loss is None:
            def output_loss(Yhat, Y): return jnp.sum((Yhat-Y)**2)/Y.shape[0]
        self.output_loss = output_loss
        self.rho_x0 = rho_x0
        self.rho_th = rho_th
        self.tau_th = tau_th
        self.tau_g = tau_g
        self.zero_coeff = zero_coeff
        self.xsat = xsat
        self.train_x0 = train_x0
        if group_lasso_fcn is not None:
            self.group_lasso_fcn = group_lasso_fcn
        if custom_regularization is not None:
            self.custom_regularization = custom_regularization
        return

    def optimization(self, adam_eta=None, adam_epochs=None, lbfgs_epochs=None, iprint=None,
                     memory=None, lbfgs_tol=None, params_min=None, params_max=None,
                     x0_min=None, x0_max=None):
        """Define optimization parameters for the system identification problem.

        Parameters
        ----------
        adam_eta : float
            Adam's learning rate (not used by LBFGS).
        adam_epochs : int
            Number of initial Adam iterations (adam_epochs=0 means pure L-BFGS-B).
        lbfgs_epochs : int
            Max number of function evaluations in the following L-BFGS iterations (lbfgs_epochs=0 means pure Adam).
        iprint : int
            How often printing L-BFGS updates (-1 = no printing, not even Adam iterations).
        memory : int
            L-BGFS memory (not used if method = 'Adam').
        lbfgs_tol : float
            Tolerance for L-BFGS-B iterations (not used if method = 'Adam').
        params_min : list of arrays, optional
            List of the same structure as self.params of lower bounds for the parameters.
        params_max : list of arrays, optional
            List of the same structure as self.params of upper bounds for the parameters.
        x0_min : array, optional
            Lower bounds for the initial state vector.
        x0_max : array, optional
            Upper bounds for the initial state vector.
        """
        self = optimization_base(self)
        if adam_eta is not None:
            self.adam_eta = adam_eta
        if adam_epochs is not None:
            self.adam_epochs = adam_epochs
        if lbfgs_epochs is not None:
            self.lbfgs_epochs = lbfgs_epochs
        if iprint is not None:
            self.iprint = iprint
        if memory is not None:
            self.memory = memory
        if lbfgs_tol is not None:
            self.lbfgs_tol = lbfgs_tol
        self.params_min = params_min
        self.params_max = params_max
        self.x0_min = x0_min
        self.x0_max = x0_max
        return

    def init(self, params=None, x0=None, sigma=0.5, seed=0):
        """Initialize model parameters

        Parameters
        ----------
        params : list of ndarrays or None
            List of arrays containing model parameters, such as params = [A,B,C,D] in case of linear systems
        x0 : ndarray or list of ndarrays or None
            Initial state vector or list of initial state vectors for multiple experiments. If None, x0=0.
        sigma : float
            Initialize matrix A=sigma*I, where I is the identity matrix
        seed : int
            Random seed for initialization (default: 0)
        """

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)

        key1, key2 = jax.random.split(jax.random.PRNGKey(seed), 2)

        if x0 is not None:
            if isinstance(x0, list):
                Nexp = len(x0)
                self.x0 = [jnp.array(x0[i]) for i in range(Nexp)]
            else:
                self.x0 = jnp.array(x0)
        else:
            self.x0 = jnp.zeros(self.nx)
            # self.x0 = 0.1 * jax.random.normal(key3, (self.nx, 1)).reshape(-1) # alternative initialization
        if self.isLinear and params is None:
            A = sigma * jnp.eye(self.nx)
            B = 0.1 * jax.random.normal(key1, (self.nx, self.nu))
            if self.y_in_x:
                self.params = [A, B]
            else:
                C = 0.1 * jax.random.normal(key2, (self.ny, self.nx))
                if not self.feedthrough:
                    self.params = [A, B, C]
                else:
                    D = jnp.zeros((self.ny, self.nu))
                    self.params = [A, B, C, D]
        else:
            if params is None:
                raise (Exception(
                    "\033[1mPlease provide the initial guess for the model parameters\033[0m"))
            else:
                self.params = [jnp.array(th) for th in params]
        self.isInitialized = True
        return

    def fit(self, Y, U):
        """Train a dynamical model using input-output data.

        Parameters
        ----------
        Y : ndarray or list of ndarrays
            Training dataset: output data. Y must be a N-by-ny numpy array
            or a list of Ni-by-ny numpy arrays, where Ni is the length of the i-th experiment.
        U : ndarray
            Training dataset: input data. U must be a N-by-nu numpy array
            or a list of Ni-by-nu numpy arrays, where Ni is the length of the i-th experiment.
        """
        nx = self.nx
        if nx < 1:
            raise (
                Exception("\033[1mModel order 'nx' must be greater than zero\033[0m"))

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)

        if not self.isInitialized:
            self.init()

        adam_epochs = self.adam_epochs
        lbfgs_epochs = self.lbfgs_epochs

        if isinstance(U, list):
            Nexp = len(U)  # number of experiments
            if not isinstance(Y, list) or len(Y) != Nexp:
                raise (Exception(
                    "\033[1mPlease provide the same number of input and output traces\033[0m"))
            for i in range(Nexp):
                U[i] = vec_reshape(U[i])
                Y[i] = vec_reshape(Y[i])
        else:
            Nexp = 1
            U = [vec_reshape(U)]
            Y = [vec_reshape(Y)]
        nu = U[0].shape[1]
        ny = Y[0].shape[1]

        if self.params is None:
            raise (Exception(
                "\033[1mPlease use the init method to initialize the parameters of the model\033[0m"))

        @jax.jit
        def SS_forward(x, u, th, sat):
            """
            Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state values in case the system is unstable.
            """
            if self.y_in_x:
                y = x[0:ny]
            else:
                y = self.output_fcn(x, u, th)
            x = self.state_fcn(x, u, th).reshape(-1)

            # saturate states to avoid numerical issues due to instability
            x = xsat(x, sat)
            return x, y

        z = self.params

        tau_th = self.tau_th
        tau_g = self.tau_g

        isL1reg = tau_th > 0
        isGroupLasso = (tau_g > 0) and (self.group_lasso_fcn is not None)
        isCustomReg = self.custom_regularization is not None

        if not isL1reg and isGroupLasso:
            tau_th = DEFAULT_SMALL_TAU_TH  # add some small L1-regularization, see Lemma 2

        self.isbounded = (self.params_min is not None) or (self.params_max is not None) or (
            self.train_x0 and ((self.x0_min is not None) or (self.x0_max is not None)))
        if self.isbounded or isL1reg or isGroupLasso:
            # define default bounds, in case they are not provided
            if self.params_min is None:
                self.params_min = list()
                for i in range(len(z)):
                    self.params_min.append(-jnp.ones_like(z[i])*np.inf)
            if self.params_max is None:
                self.params_max = list()
                for i in range(len(z)):
                    self.params_max.append(jnp.ones_like(z[i])*np.inf)
            if self.train_x0:
                if self.x0_min is None:
                    self.x0_min = [-jnp.ones_like(self.x0)*np.inf]*Nexp
                if not isinstance(self.x0_min, list):
                    # repeat the same initial-state bound on all experiments
                    self.x0_min = [self.x0_min]*Nexp
                if len(self.x0_min) is not Nexp:
                    # number of experiments has changed, repeat the same initial-state bound on all experiments
                    self.x0_min = [self.x0_min[0]]*Nexp
                if self.x0_max is None:
                    self.x0_max = [jnp.ones_like(self.x0)*np.inf]*Nexp
                if not isinstance(self.x0_max, list):
                    self.x0_max = [self.x0_max]*Nexp
                if len(self.x0_max) is not Nexp:
                    self.x0_max = [self.x0_max[0]]*Nexp

        x0 = self.x0
        if x0 is None:
            x0 = [jnp.zeros(nx)]*Nexp
        elif not isinstance(x0, list):
            x0 = [x0]*Nexp

        def train_model(solver, solver_iters, z, x0, J0):
            """
            Train a state-space model using the specified solver.

            Parameters
            ----------
            solver : str
                The solver to use for optimization.
            solver_iters : int
                The maximum number of iterations for the solver.
            z : list
                The initial guess for the model parameters.
            x0 : ndarray or list of ndarrays
                The initial guess for the initial state vector, or a list of initial state vectors for multiple experiments.
            J0 : float
                The initial cost value.

            Returns
            -------
            tuple
                A tuple containing the updated model parameters, initial state vector, final cost value, and number of iterations.
            """
            if solver_iters == 0:
                return z, x0, J0, 0.

            nth = len(z)
            if solver == "LBFGS" and (isGroupLasso or isL1reg):
                # duplicate params to create positive and negative parts
                z.extend(z)
                for i in range(nth):
                    zi = z[i].copy()
                    # we could also consider bounds here, if present
                    z[i] = jnp.maximum(zi, 0.)+EPSIL_LASSO
                    z[nth+i] = -jnp.minimum(zi, 0.)+EPSIL_LASSO

            nzmx0 = len(z)
            if self.train_x0:
                for i in range(Nexp):
                    # one initial state per experiment
                    z.append(x0[i].reshape(-1))
                # in case of group-Lasso, if state #i is removed from A,B,C then the corresponding x0(i)=0
                # because of L2-regularization on x0.

            # total number of optimization variables
            nvars = sum([zi.size for zi in z])

            @jax.jit
            def loss(th, x0):
                """
                Calculate the loss function for system identification.

                Parameters
                ----------
                th : array
                    The system parameters.
                x0 : array
                    The initial state or list of initial states.

                Returns
                -------
                float
                    The loss value.
                """
                f = partial(SS_forward, th=th, sat=self.xsat)
                cost = 0.
                for i in range(Nexp):
                    _, Yhat = jax.lax.scan(f, x0[i], U[i])
                    cost += self.output_loss(Yhat, Y[i])
                return cost

            t_solve = time.time()

            if solver == "Adam":
                if self.train_x0:
                    @jax.jit
                    def J(z):
                        th = z[:nzmx0]
                        x0 = z[nzmx0:]
                        cost = loss(th, x0) + self.rho_x0 * \
                            sum([jnp.sum(x0i**2)
                                for x0i in x0]) + self.rho_th*l2reg(th)
                        if isL1reg:
                            cost += tau_th*l1reg(th)
                        if isGroupLasso:
                            cost += tau_g*self.group_lasso_fcn(th, x0)
                        if isCustomReg:
                            cost += self.custom_regularization(th, x0)
                        return cost
                else:
                    @jax.jit
                    def J(z):
                        cost = loss(z, x0) + self.rho_th*l2reg(z)
                        if isL1reg:
                            cost += tau_th*l1reg(z)
                        if isGroupLasso:
                            cost += tau_g*self.group_lasso_fcn(z, x0)
                        if isCustomReg:
                            cost += self.custom_regularization(z, x0)
                        return cost

                def JdJ(z):
                    return jax.value_and_grad(J)(z)

                lb = None
                ub = None
                if self.isbounded:
                    lb = list(self.params_min)
                    ub = list(self.params_max)
                    if self.train_x0:
                        lb.extend(self.x0_min)
                        ub.extend(self.x0_max)

                z, Jopt = adam_solver(
                    JdJ, z, solver_iters, self.adam_eta, self.iprint, lb, ub)

            elif solver == "LBFGS":
                # L-BFGS-B params (no L1 regularization)
                options = lbfgs_options(
                    min(self.iprint, 90), solver_iters, self.lbfgs_tol, self.memory)

                # if self.iprint > -1:
                #     print(
                #         "Solving NLP with L-BFGS (%d optimization variables) ..." % nvars)

                if isGroupLasso or isL1reg:
                    bounds = get_bounds(
                        z[0:nth], EPSIL_LASSO, self.params_min, self.params_max)
                    if self.train_x0:
                        bounds[0].extend(self.x0_min)
                        bounds[1].extend(self.x0_max)

                if not isGroupLasso:
                    if not isL1reg:
                        if self.train_x0:
                            @jax.jit
                            def J(z):
                                th = z[:nzmx0]
                                x0 = z[nzmx0:]
                                cost = loss(th, x0) + self.rho_x0 * \
                                    sum([jnp.sum(x0i**2)
                                        for x0i in x0]) + self.rho_th*l2reg(th)
                                if isCustomReg:
                                    cost += self.custom_regularization(th, x0)
                                return cost
                        else:
                            @jax.jit
                            def J(z):
                                cost = loss(z, x0) + self.rho_th*l2reg(z)
                                if isCustomReg:
                                    cost += self.custom_regularization(z, x0)
                                return cost
                        cb, cb_final = lbfgs_callback(solver_iters, options["iprint"], J)
                        options.pop("iprint", None)
                        options["disp"] = False
                        if not self.isbounded:
                            solver = jaxopt.ScipyMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options,  callback=cb)
                            z, state = solver.run(z)
                        else:
                            lb = list(self.params_min)
                            ub = list(self.params_max)
                            if self.train_x0:
                                lb.extend(self.x0_min)
                                ub.extend(self.x0_max)
                            solver = jaxopt.ScipyBoundedMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options, callback=cb)
                            z, state = solver.run(z, bounds=(lb, ub))
                        if cb_final is not None:
                            cb_final(state.fun_val)
                        iter_num = state.iter_num
                        Jopt = state.fun_val
                    else:
                        # Optimize wrt to split positive and negative part of model parameters
                        if self.train_x0:
                            @jax.jit
                            def J(z):
                                x0 = z[nzmx0:]
                                th = [z1 - z2 for (z1, z2)
                                      in zip(z[0:nth], z[nth:2 * nth])]
                                cost = loss(th, x0) + self.rho_x0 * sum([jnp.sum(x0i**2) for x0i in x0]) + self.rho_th*l2reg(z[0:nth]) + self.rho_th*l2reg(
                                    z[nth:2 * nth]) + tau_th*linreg(z[0:nth]) + tau_th*linreg(z[nth:2 * nth])
                                if isCustomReg:
                                    cost += self.custom_regularization(th, x0)
                                return cost
                        else:
                            @jax.jit
                            def J(z):
                                th = [z1 - z2 for (z1, z2)
                                      in zip(z[0:nth], z[nth:2 * nth])]
                                cost = loss(th, x0) + self.rho_th*l2reg(z[0:nth]) + self.rho_th*l2reg(
                                    z[nth:2 * nth]) + tau_th*linreg(z[0:nth]) + tau_th*linreg(z[nth:2 * nth])
                                if isCustomReg:
                                    cost += self.custom_regularization(th, x0)
                                return cost

                        cb, cb_final = lbfgs_callback(solver_iters, options["iprint"], J)
                        options.pop("iprint", None)
                        options["disp"] = False
                        solver = jaxopt.ScipyBoundedMinimize(
                            fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options, callback=cb)
                        z, state = solver.run(z, bounds=bounds)
                        if cb_final is not None:
                            cb_final(state.fun_val)
                        z[0:nth] = [
                            z1 - z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])]
                        iter_num = state.iter_num
                        Jopt = state.fun_val

                else:  # group Lasso
                    if self.train_x0:
                        @jax.jit
                        def J(z):
                            x0 = z[nzmx0:]
                            th = [z1 - z2 for (z1, z2)
                                  in zip(z[0:nth], z[nth:2 * nth])]
                            cost = loss(th, x0) + self.rho_x0 * sum([jnp.sum(x0i**2) for x0i in x0]) + self.rho_th*l2reg(z[0:nth]) + self.rho_th*l2reg(
                                z[nth:2 * nth]) + tau_th*linreg(z[0:nth]) + tau_th*linreg(z[nth:2 * nth])
                            if tau_g > 0:
                                cost += tau_g * \
                                    self.group_lasso_fcn(
                                        [z1 + z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])], x0)
                            if isCustomReg:
                                cost += self.custom_regularization(th, x0)
                            return cost
                    else:
                        @jax.jit
                        def J(z):
                            th = [z1 - z2 for (z1, z2)
                                  in zip(z[0:nth], z[nth:2 * nth])]
                            cost = loss(th, x0) + self.rho_th*l2reg(z[0:nth]) + self.rho_th*l2reg(
                                z[nth:2 * nth]) + tau_th*linreg(z[0:nth]) + tau_th*linreg(z[nth:2 * nth])
                            if tau_g > 0:
                                cost += tau_g * \
                                    self.group_lasso_fcn(
                                        [z1 + z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])], x0)
                            if isCustomReg:
                                cost += self.custom_regularization(th, x0)
                            return cost

                    cb, cb_final = lbfgs_callback(solver_iters, options["iprint"], J)
                    options.pop("iprint", None)
                    options["disp"] = False
                    solver = jaxopt.ScipyBoundedMinimize(
                        fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options, callback=cb)
                    z, state = solver.run(z, bounds=bounds)
                    if cb_final is not None:
                        cb_final(state.fun_val)
                    z[0:nth] = [
                        z1 - z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])]
                    iter_num = state.iter_num
                    Jopt = state.fun_val

                print(' L-BFGS-B done in %d iterations.' % iter_num)

            else:
                raise (Exception("\033[1mUnknown solver\033[0m"))

            if self.train_x0:
                x0 = [z[nzmx0+i].reshape(-1) for i in range(Nexp)]

            t_solve = time.time() - t_solve
            return z[0:nth], x0, Jopt, t_solve

        z, x0, Jopt, t_solve1 = train_model('Adam', adam_epochs, z, x0, np.inf)
        z, x0, Jopt, t_solve2 = train_model('LBFGS', lbfgs_epochs, z, x0, Jopt)
        t_solve = t_solve1+t_solve2

        x0 = [np.array(x0i) for x0i in x0]

        if not self.isbounded:
            # reset original bounds, possibly altered in case of L1-regularization or group-Lasso and L-BFGS is used
            self.params_min = None
            self.params_max = None
            self.x0_min = None
            self.x0_max = None

        if self.isLinear:
            # Change state-space realization by re-ordering the states as a function of the norm of the rows of A
            A = np.array(z[0])
            ii = np.argsort(-np.sum(A**2, 1))
            A = A[ii, :]
            A = A[:, ii]
            B = np.array(z[1])[ii, :]
            if not self.y_in_x:
                C = np.array(z[2])[:, ii]
                if self.feedthrough:
                    D = np.array(z[3])
            x0 = [x0i[ii] for x0i in x0]  # initial state

            # Zero coefficients smaller than zero_coeff in absolute value
            A[np.abs(A) <= self.zero_coeff] = 0.
            B[np.abs(B) <= self.zero_coeff] = 0.
            if not self.y_in_x:
                C[np.abs(C) <= self.zero_coeff] = 0.
                if self.feedthrough:
                    D[np.abs(D) <= self.zero_coeff] = 0.
                    z = [A, B, C, D]
                else:
                    D = np.zeros((ny, nu))
                    z = [A, B, C]
            else:
                C = np.hstack((np.eye(ny), np.zeros((ny, nx-ny))))
                D = np.zeros((ny, nu))
                z = [A, B]
        else:
            # Zero coefficients smaller than zero_coeff in absolute value
            for i in range(len(z)):
                z[i] = np.array(z[i])
                z[i][np.abs(z[i]) <= self.zero_coeff] = 0.

        self.params = z

        # Zero coefficients smaller than zero_coeff in absolute value
        for i in range(Nexp):
            x0[i][np.abs(x0[i]) <= self.zero_coeff] = 0.0

        # Check if state saturation is active.
        for i in range(Nexp):
            # this overrides possible predict() methods defined in subclasses
            _, Xt = Model.predict(self, x0[i], U[i], 0, 0)
            sat_activated = np.any(Xt > self.xsat) or np.any(Xt < -self.xsat)
            if sat_activated:
                print(
                    "\033[1mWarning: state saturation is active at the solution. \nYou may have to increase the values of 'xsat' or 'rho_x0'\033[0m")

        # Check model sparsity
        sparsity = dict()
        if self.isLinear:
            sparsity["A"] = 100*(1.-np.sum(np.abs(A) > self.zero_coeff)/A.size)
            sparsity["B"] = 100*(1.-np.sum(np.abs(B) > self.zero_coeff)/B.size)
            sparsity["C"] = 100*(1.-np.sum(np.abs(C) > self.zero_coeff)/C.size)
            sparsity["D"] = 100*(1.-np.sum(np.abs(D) > self.zero_coeff)/D.size)
            removable = list()
            for i in range(nx):
                if np.sum(np.abs(A[i, :]) <= self.zero_coeff) == nx and np.sum(np.abs(A[:, i]) <= self.zero_coeff) == nx \
                        and np.sum(np.abs(B[i, :]) <= self.zero_coeff) == nu and np.sum(np.abs(C[:, i]) <= self.zero_coeff) == ny:
                    # print("state #%2d can be eliminated" % (i+1))
                    removable.append(i)
            sparsity["removable_states"] = removable
            removable = list()
            for i in range(nu):
                if np.sum(np.abs(B[:, i]) <= self.zero_coeff) == nx and np.sum(np.abs(D[:, i]) <= self.zero_coeff) == ny:
                    # print("input #%2d can be eliminated" % (i+1))
                    removable.append(i)
            sparsity["removable_inputs"] = removable

        if self.isqLPV:
            removable = list()
            _, _, _, _, Ap, Bp, Cp, Dp = self.ssdata()
            for i in range(self.nqLPVpar):
                if np.max(np.abs(Ap[i])) <= self.zero_coeff and np.max(np.abs(Bp[i])) <= self.zero_coeff and np.max(np.abs(Cp[i])) <= self.zero_coeff and np.max(np.abs(Dp[i])) <= self.zero_coeff:
                    # print("parameter #%2d can be eliminated" % (i+1))
                    removable.append(i)
            sparsity["removable_parameters"] = removable

        sparsity["nonzero_parameters"] = [np.sum([np.sum(np.abs(z[i]) > self.zero_coeff) for i in range(
            len(z))]), np.sum([z[i].size for i in range(len(z))])]

        self.x0 = x0
        if Nexp == 1:
            self.x0 = self.x0[0]
        self.Jopt = Jopt
        self.t_solve = t_solve
        self.Nexp = Nexp
        self.sat_activated = sat_activated
        self.sparsity = sparsity
        return

    def parallel_fit(self, Y, U, init_fcn, seeds, n_jobs=None):
        """
        Fits the model in parallel using multiple seeds.

        Parameters:
            Y : ndarray or list of ndarrays
                Training dataset: output data. Y must be a N-by-ny numpy array
                or a list of Ni-by-ny numpy arrays, where Ni is the length of the i-th experiment.
            U : ndarray
                Training dataset: input data. U must be a N-by-nu numpy array
                or a list of Ni-by-nu numpy arrays, where Ni is the length of the i-th experiment.
            init_fcn (callable): A function that initializes the model parameters given a seed.
            seeds (array-like): The seeds used for initialization.
            n_jobs (int): The number of parallel jobs to run (default is None, which means using all available cores).

        Returns:
            list: A list of fitted models.
        """
        def single_fit(seed):
            if not jax.config.jax_enable_x64:
                # Enable 64-bit computations
                jax.config.update("jax_enable_x64", True)
            self.init(params=init_fcn(seed))
            # if self.iprint > -1:
            #     print(
            #         "\033[1m" + f"Fitting model with seed = {seed} ... " + "\033[0m")
            self.fit(Y, U)
            # if self.iprint > -1:
            #     print("\033[1m" + f"Seed = {seed}: done." + "\033[0m")
            return self
        if n_jobs is None:
            n_jobs = cpu_count()  # all available CPUs
        return Parallel(n_jobs=n_jobs)(delayed(single_fit)(seed) for seed in seeds)

    def learn_x0(self, U, Y, rho_x0=None, RTS_epochs=1, verbosity=True, LBFGS_refinement=False,
                 LBFGS_rho_x0=1.e-8, lbfgs_epochs=1000, Q=None, R=None):
        """Estimate x0 by Rauch–Tung–Striebel smoothing (Sarkka and Svenson, 2023, p.268),
        possibly followed by L-BFGS optimization.

        (C) 2023 A. Bemporad

        Parameters
        ----------
        U : ndarray
            Input data, U must be a N-by-nu numpy array
        Y : ndarray
            Output data, Y must be a N-by-ny numpy array
        rho_x0 : float
            L2-regularization on initial state x0, 0.5*rho_x0*||x0||_2^2 (default: model.rho_x0)
        RTS_epochs : int
            Number of forward KF and backward RTS passes
        verbosity : bool
            If false, removes printout of operations
        LBFGS_refinement : bool
            If True, refine solution via L-BFGS optimization. Also used in the case bounds on x0 have been specified and the value of x0 estimated by RTS is not feasible.
        LBFGS_rho_x0 : float
            L2-regularization used by L-BFGS, by default 1.e-8
        lbfgs_epochs : int
            Max number of L-BFGS iterations
        Q : ndarray
            Process noise covariance matrix, by default 1.e-5*I. Matrix Q could be set smaller (e.g., 1.e-8*I),although in the case of very good models the covariance matrix of the state estimation error may become very small and RTS smoothing numerically unstable, and higher values of Q should be used.
        R : ndarray
            Measurement noise covariance matrix, by default the identity matrix I

        Returns
        -------
        array
            Optimal initial state x0.
        """
        nx = self.nx
        ny = self.ny
        N = U.shape[0]

        isLinear = self.isLinear
        if isLinear:
            A_LTI, _, C_LTI, _ = self.params2ABCD()
        else:
            @jax.jit
            def Ck(x, u):
                return jax.jacrev(self.output_fcn)(x, u=u, params=self.params)

            @jax.jit
            def Ak(x, u):
                return jax.jacrev(self.state_fcn)(x, u=u, params=self.params)

        if rho_x0 is None:
            rho_x0 = self.rho_x0
        if R is None:
            R = np.eye(ny)
        if Q is None:
            Q = 1.e-5 * np.eye(nx)

        # Forward EKF pass:
        @jax.jit
        def EKF_update(state, yuk):
            x, P, mse_loss = state
            yk = yuk[:ny]
            u = yuk[ny:]

            # measurement update
            y = self.output_fcn(x, u, self.params)
            if not isLinear:
                Ckk = Ck(x, u)
            else:
                Ckk = C_LTI
            PC = P @ Ckk.T
            # M = PC / (R + C @PC) # this solves the linear system M*(R + C @PC) = PC
            # Note: Matlab's mrdivide A / B = (B'\A')' = np.linalg.solve(B.conj().T, A.conj().T).conj().T
            M = jax.scipy.linalg.solve((R+Ckk@PC), PC.T, assume_a='pos').T
            e = yk-y
            mse_loss += np.sum(e**2)  # just for monitoring purposes
            x1 = x + M@e  # x(k | k)

            # Standard Kalman measurement update
            # P -= M@PC.T
            # P = (P + P.T)/2. # P(k|k)

            # Joseph stabilized covariance update
            IKH = -M@Ckk
            IKH += jnp.eye(nx)
            P1 = IKH@P@IKH.T+M@R@M.T  # P(k|k)

            # Time update
            if not isLinear:
                Akk = Ak(x1, u)
            else:
                Akk = A_LTI
            P2 = Akk@P1@Akk.T+Q
            # P2 = (P2+P2.T)/2.
            x2 = self.state_fcn(x1, u, self.params)
            if not isLinear:
                output = (x1, P1, x2, P2, Akk)
            else:
                # Avoid returning the same A matrix everytime
                output = (x1, P1, x2, P2)

            return (x2, P2, mse_loss), output

        @jax.jit
        def RTS_update(state, input):
            x, P = state
            if not isLinear:
                P1, P2, x1, x2, A = input
            else:
                P1, P2, x1, x2 = input  # The A matrix is the LTI state-update matrix defined earlier
                A = A_LTI

            # G=(PP1[k]@AA[k].T)/PP2[k]
            try:
                G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='pos').T
            except:
                G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='gen').T
            x = x1+G@(x-x2)
            P = P1+G@(P-P2)@G.T
            return (x, P), None

        # L2-regularization on initial state x0, 0.5*rho_x0*||x0||_2^2
        P = np.eye(nx) / (rho_x0 * N)
        x = np.zeros(nx)

        for epoch in range(RTS_epochs):
            mse_loss = 0.

            # Forward EKF pass
            state = (x, P, mse_loss)
            state, output = jax.lax.scan(EKF_update, state, np.hstack((Y, U)))
            if not isLinear:
                XX1, PP1, XX2, PP2, AA = output
            else:
                XX1, PP1, XX2, PP2 = output
            # PP1 = P(k | k)
            # PP2 = P(k + 1 | k)
            # XX1 = x(k | k)
            # XX2 = x(k + 1 | k)
            mse_loss = state[2]/N

            # RTS smoother pass:
            x = XX2[N-1]
            P = PP2[N-1]
            state = (x, P)
            if not isLinear:
                input = (PP1[::-1], PP2[::-1], XX1[::-1], XX2[::-1], AA[::-1])
            else:
                input = (PP1[::-1], PP2[::-1], XX1[::-1], XX2[::-1])
            state, _ = jax.lax.scan(RTS_update, state, input)
            x, P = state

            if verbosity:
                _bar_width = 20
                frac = (epoch + 1) / RTS_epochs
                filled = int(_bar_width * frac)
                bar = '█' * filled + '░' * (_bar_width - filled)
                print(f"\rRTS smoothing: {100*frac:3.0f}%|{bar}| {epoch+1}/{RTS_epochs} [MSE={float(mse_loss):.4e}]",
                      end='\033[K', flush=True)

        if verbosity:
            print()

        x = np.array(x)

        isstatebounded = self.x0_min is not None or self.x0_max is not None
        if isstatebounded:
            lb = self.x0_min
            if isinstance(lb, list):
                lb = lb[0]
            if lb is None:
                lb = -np.inf*np.ones(nx)
            ub = self.x0_max
            if isinstance(ub, list):
                ub = ub[0]
            if ub is None:
                ub = np.inf*np.ones(nx)
            if np.any(x < lb) or np.any(x > ub):
                LBFGS_refinement = True

        if LBFGS_refinement:
            # Refine via L-BFGS with very small penalty on x0
            options = lbfgs_options(
                iprint=-1, iters=lbfgs_epochs, lbfgs_tol=1.e-10, memory=100)

            @jax.jit
            def SS_step(x, u):
                y = self.output_fcn(x, u, self.params)
                x = self.state_fcn(x, u, self.params).reshape(-1)
                return x, y

            @jax.jit
            def J(x0):
                _, Yhat = jax.lax.scan(SS_step, x0, U)
                return jnp.sum((Yhat - Y) ** 2) / U.shape[0]+.5*LBFGS_rho_x0*jnp.sum(x0**2)
            cb, cb_final = lbfgs_callback(options["maxfun"], options["iprint"], J)
            options.pop("iprint", None)
            options["disp"] = False
            if not isstatebounded:
                solver = jaxopt.ScipyMinimize(
                    fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options, callback=cb)
                x, state = solver.run(x)
            else:
                solver = jaxopt.ScipyBoundedMinimize(
                    fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options, callback=cb)
                x, state = solver.run(x, bounds=(lb, ub))
            if cb_final is not None:
                cb_final(state.fun_val)
            x = np.array(x)

            if verbosity:
                mse_loss = state.fun_val-.5*LBFGS_rho_x0*np.sum(x**2)
                print(f"Final MSE (after L-BFGS refinement) = {mse_loss:.4e}")
        return x

    def sparsity_analysis(self):
        line = "-"*50 + "\n"
        txt = "Model sparsity:\n" + line
        if self.isLinear:
            A, B, C, D = self.params2ABCD()
            txt += "  A: %6.2f%%" % self.sparsity["A"]
            txt += " (zero parameters: %d/%d)\n" % (np.sum(np.abs(A)
                                                           <= self.zero_coeff), A.size)
            txt += "  B: %6.2f%%" % self.sparsity["B"]
            txt += " (zero parameters: %d/%d)\n" % (np.sum(np.abs(B)
                                                           <= self.zero_coeff), B.size)
            txt += "  C: %6.2f%%" % self.sparsity["C"]
            txt += " (zero parameters: %d/%d)\n" % (np.sum(np.abs(C)
                                                           <= self.zero_coeff), C.size)
            txt += "  D: %6.2f%%" % self.sparsity["D"]
            txt += " (zero parameters: %d/%d)\n" % (np.sum(np.abs(D)
                                                           <= self.zero_coeff), D.size)
            txt += line + "Removable states: "
            if len(self.sparsity['removable_states']) == 0:
                txt += "none"
            else:
                txt += ', '.join(str(n)
                                 for n in self.sparsity['removable_states'])
            txt += "\nRemovable inputs: "
            if len(self.sparsity['removable_inputs']) == 0:
                txt += "none"
            else:
                txt += ', '.join(str(n)
                                 for n in self.sparsity['removable_inputs'])
        else:
            txt += "%d nonzero model parameters out of %d (%6.2f%% sparsity)" % (
                self.sparsity["nonzero_parameters"][0], self.sparsity["nonzero_parameters"][1], 100*(1.-self.sparsity["nonzero_parameters"][0]/self.sparsity["nonzero_parameters"][1]))
        txt += "\n" + line
        return txt

    def dcgain_loss(self, Uss, Yss, beta=1.):
        """
        Define a loss function for fitting the DC gain of the model to given steady-state input-output pairs.

        Parameters:
        -----------
        Uss : array-like
            Steady-state input values.
        Yss : array-like
            Steady-state output values.
        beta : float, optional
            Penalty on the DC gain loss

        Returns:
        --------
        dcgain_loss : function
            A function that computes the DC gain loss given model parameters. This function can be used a custom regularization loss in the fit method.

        Notes:
        ------
        The DC gain loss is computed by comparing the predicted steady-state outputs of the model
        with the actual steady-state outputs provided. The steady-state is determined by solving
        the steady-state equations for the system, either using a Broyden solver for
        nonlinear systems or a direct solution for linear systems.
        """

        @jax.jit
        def ss_residual(x, other):
            u, params = other
            xnext = self.state_fcn(x, jnp.array(
                u).reshape(self.nu), params).reshape(-1, 1)
            return (xnext-x.reshape(-1, 1)).ravel()

        @jax.jit
        def steady_state(uss, params):
            # Solve steady-state equations to get xss such that xss = f(xss,uss)
            if not self.isLinear:
                xss = jnp.zeros(self.nx)  # initial guess
                broyden = jaxopt.Broyden(fun=ss_residual, tol=1.e-6, stepsize=1.0) # linesearch is ignored when stepsize > 0
                xss = broyden.run(other=[uss, params], init_params=xss).params
            else:
                A = params[0]
                B = params[1]
                xss = jnp.linalg.solve(jnp.eye(self.nx)-A, B@uss)
            yss = self.output_fcn(xss, uss, params)
            return yss

        # Loss function: single sample
        @jax.jit
        def dcgain_loss_k(uss, yss, params):
            yss_hat = steady_state(uss, params)
            return jnp.sum((yss-yss_hat)**2)

        # Loss function for unit-dc-gain: multiple samples
        dcgain_loss_vmap = jax.jit(
            jax.vmap(dcgain_loss_k, in_axes=(0, 0, None)))

        def dcgain_loss(params, x0):
            return beta*jnp.sum(dcgain_loss_vmap(Uss, Yss, params))/Yss.shape[0]

        return dcgain_loss
