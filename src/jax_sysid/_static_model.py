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
from joblib import Parallel, delayed, cpu_count
from jax_sysid.utils import lbfgs_options, vec_reshape
from ._solvers import (
    EPSIL_LASSO, DEFAULT_SMALL_TAU_TH,
    l2reg, l1reg, linreg,
    optimization_base, adam_solver, lbfgs_callback,
    get_bounds,
)


class StaticModel(object):
    """
    Base class of trainable static input/output models
    """

    def __init__(self, ny, nu, output_fcn=None):
        """
        Initialize model structure.

        ny (int) : number of outputs

        nu (int) : number of inputs

        output_fcn (function) : function handle to the output function y(k)=output_fcn(u(k),params).
        The function must have the signature f(u, params) where u is the input and params is a list of ndarrays of parameters. The function must be vectorized with respect to u, i.e., it must be able to handle a matrix of inputs u, one row per input value. The function must return a matrix of outputs y, where each row corresponds to an input in u.
        """

        self.ny = ny  # number of outputs
        self.nu = nu  # number of inputs
        self.output_fcn = output_fcn

        self.loss()  # define default loss function
        self.optimization()  # define default optimization parameters
        self.isInitialized = False  # model parameters are not initialized yet

        self.Jopt = None
        self.t_solve = None
        self.sparsity = None
        self.group_lasso_fcn = None
        self.custom_regularization = None

    def predict(self, U):
        """
        Evaluate the static model as a function of the input.

        Parameters
        ----------
        U : ndarray
            Input sequence.

        Returns
        -------
        Y : ndarray
            Output signal.
        """
        U = vec_reshape(U)
        return self.output_fcn(U, self.params)

    def loss(self, output_loss=None, rho_th=0.01, tau_th=0.0, tau_g=0.0, group_lasso_fcn=None, zero_coeff=0., custom_regularization=None):
        """Define the overall loss function for training the model.

        Parameters
        ----------
        output_loss : function
            Loss function penalizing output fit errors, loss=output_loss(Yhat,Y), where Yhat is the sequence of predicted outputs and Y is the measured output.
            If None, use standard mean squared error loss=sum((Yhat-Y)**2)/Y.shape[0]
        rho_th : float
            L2-regularization on model parameters
        tau_th : float
            L1-regularization on model parameters
        tau_g : float
            group-Lasso regularization penalty
        group_lasso_fcn : function
            function f(params) defining the group-Lasso penalty on the model parameters "params", minimized as tau_g*sum(||params_i||_2).
        zero_coeff : _type_
            Entries smaller than zero_coeff are set to zero. Useful when tau_th>0 or tau_g>0.
        custom_regularization : function
            Additional custom regularization term, a function of the model parameters
            custom_regularization(params).
        """
        if output_loss is None:
            def output_loss(Yhat, Y): return jnp.sum((Yhat - Y)**2)/Y.shape[0]
        self.output_loss = output_loss
        self.rho_th = rho_th
        self.tau_th = tau_th
        self.tau_g = tau_g
        self.zero_coeff = zero_coeff
        if group_lasso_fcn is not None:
            self.group_lasso_fcn = group_lasso_fcn
        if custom_regularization is not None:
            self.custom_regularization = custom_regularization
        return

    def optimization(self, adam_eta=None, adam_epochs=None, lbfgs_epochs=None, iprint=None, memory=None, lbfgs_tol=None, params_min=None, params_max=None):
        """Define optimization parameters for the training problem.

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
        lbfsg_tol : float
            Tolerance for L-BFGS-B iterations (not used if method = 'Adam').
        params_min : list of arrays, optional
            List of the same structure as self.params of lower bounds for the parameters.
        params_max : list of arrays, optional
            List of the same structure as self.params of upper bounds for the parameters.
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
        return

    def init(self, params=None):
        """Initialize model parameters

        Parameters
        ----------
        params : list of ndarrays or None
            List of arrays containing model parameters
        """
        if params is None:
            raise (Exception(
                "\033[1mPlease provide the initial guess for the model parameters\033[0m"))
        else:
            self.params = [jnp.array(th) for th in params]
        self.isInitialized = True
        return

    def fit(self, Y, U):
        """
        Parameters
        ----------
        Y : ndarray
            Training dataset: output data. Y must be a N-by-ny numpy array.
        U : ndarray
            Training dataset: input data. U must be a N-by-nu numpy array.
        """

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)

        if not self.isInitialized:
            self.init()

        adam_epochs = self.adam_epochs
        lbfgs_epochs = self.lbfgs_epochs

        U = vec_reshape(U)
        Y = vec_reshape(Y)

        if self.params is None:
            raise (Exception(
                "\033[1mPlease use the init method to initialize the parameters of the model\033[0m"))

        z = self.params

        if self.params_min is not None and self.params_max is None:
            self.params_max = list()
            for i in range(len(z)):
                self.params_max.append(jnp.ones_like(z[i])*np.inf)
        if self.params_max is not None and self.params_min is None:
            self.params_min = list()
            for i in range(len(z)):
                self.params_min.append(-jnp.ones_like(z[i])*np.inf)

        tau_th = self.tau_th
        tau_g = self.tau_g

        isL1reg = tau_th > 0
        isGroupLasso = (tau_g > 0) and (self.group_lasso_fcn is not None)
        isCustomReg = (self.custom_regularization is not None)

        if not isL1reg and isGroupLasso:
            tau_th = DEFAULT_SMALL_TAU_TH

        def train_model(solver, solver_iters, z, J0):
            """
            Train a static model using the specified solver.

            Parameters
            ----------
            solver : str
                The solver to use for optimization.
            solver_iters : int
                The maximum number of iterations for the solver.
            z : list
                The initial guess for the model parameters.
            J0 : float
                The initial cost value.

            Returns
            -------
            tuple
                A tuple containing the updated model parameters, final cost value, and number of iterations.
            """
            if solver_iters == 0:
                return z, J0, 0.

            nth = len(z)
            if solver == "LBFGS" and (isGroupLasso or isL1reg):
                # duplicate params to create positive and negative parts
                z.extend(z)
                for i in range(nth):
                    zi = z[i].copy()
                    z[i] = jnp.maximum(zi, 0)+EPSIL_LASSO
                    z[nth+i] = -jnp.minimum(zi, 0)+EPSIL_LASSO

            # total number of optimization variables
            nvars = sum([zi.size for zi in z])

            @jax.jit
            def loss(th):
                """
                Calculate the loss function.

                Parameters
                ----------
                th : array
                    The system parameters.

                Returns
                -------
                float
                    The loss value.
                """
                Yhat = self.output_fcn(U, th).reshape(-1, self.ny)
                cost = self.output_loss(Yhat, Y)
                return cost

            t_solve = time.time()

            if solver == "Adam":
                @jax.jit
                def J(z):
                    cost = loss(z) + self.rho_th*l2reg(z)
                    if isL1reg:
                        cost += tau_th*l1reg(z)
                    if isGroupLasso:
                        cost += tau_g*self.group_lasso_fcn(z)
                    if isCustomReg:
                        cost += self.custom_regularization(z)
                    return cost

                def JdJ(z):
                    return jax.value_and_grad(J)(z)

                z, Jopt = adam_solver(
                    JdJ, z, solver_iters, self.adam_eta, self.iprint, self.params_min, self.params_max)

            elif solver == "LBFGS":
                # L-BFGS-B params (no L1 regularization)
                options = lbfgs_options(
                    min(self.iprint, 90), solver_iters, self.lbfgs_tol, self.memory)

                # if self.iprint > -1:
                #     print(
                #         "Solving NLP with L-BFGS (%d optimization variables) ..." % nvars)

                if not isGroupLasso:
                    if not isL1reg:
                        def J(z):
                            cost = loss(z) + self.rho_th*l2reg(z)
                            if isCustomReg:
                                cost += self.custom_regularization(z)
                            return cost

                        cb, cb_final = lbfgs_callback(solver_iters, options["iprint"], J)
                        options.pop("iprint", None)
                        options["disp"] = False
                        if (self.params_min is None) and (self.params_max is None):
                            solver = jaxopt.ScipyMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options, callback=cb)
                            z, state = solver.run(z)
                        else:
                            solver = jaxopt.ScipyBoundedMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options, callback=cb)
                            z, state = solver.run(z, bounds=(
                                self.params_min, self.params_max))
                        if cb_final is not None:
                            cb_final(state.fun_val)
                        iter_num = state.iter_num
                        Jopt = state.fun_val
                    else:
                        # Optimize wrt to split positive and negative part of model parameters
                        @jax.jit
                        def J(z):
                            th = [z1 - z2 for (z1, z2)
                                  in zip(z[0:nth], z[nth:2 * nth])]
                            cost = loss(th) + self.rho_th*l2reg(z[0:nth]) + self.rho_th*l2reg(
                                z[nth:2 * nth]) + tau_th*linreg(z[0:nth]) + tau_th*linreg(z[nth:2 * nth])
                            if isCustomReg:
                                cost += self.custom_regularization(th)
                            return cost

                        cb, cb_final = lbfgs_callback(solver_iters, options["iprint"], J)
                        options.pop("iprint", None)
                        options["disp"] = False
                        solver = jaxopt.ScipyBoundedMinimize(
                            fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options, callback=cb)

                        bounds = get_bounds(
                            z[0:nth], EPSIL_LASSO, self.params_min, self.params_max)
                        z, state = solver.run(z, bounds=bounds)
                        if cb_final is not None:
                            cb_final(state.fun_val)
                        z[0:nth] = [
                            z1 - z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])]
                        iter_num = state.iter_num
                        Jopt = state.fun_val

                else:  # group Lasso
                    @jax.jit
                    def J(z):
                        th = [z1 - z2 for (z1, z2)
                              in zip(z[0:nth], z[nth:2 * nth])]
                        cost = loss(th) + self.rho_th*l2reg(z[0:nth]) + self.rho_th*l2reg(
                            z[nth:2 * nth]) + tau_th*linreg(z[0:nth]) + tau_th*linreg(z[nth:2 * nth])
                        if tau_g > 0:
                            cost += tau_g * \
                                self.group_lasso_fcn(
                                    [z1 + z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])])
                        if isCustomReg:
                            cost += self.custom_regularization(th)
                        return cost

                    cb, cb_final = lbfgs_callback(solver_iters, options["iprint"], J)
                    options.pop("iprint", None)
                    options["disp"] = False
                    solver = jaxopt.ScipyBoundedMinimize(
                        fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options, callback=cb)

                    bounds = get_bounds(
                        z[0:nth], EPSIL_LASSO, self.params_min, self.params_max)
                    z, state = solver.run(z, bounds=bounds)
                    if cb_final is not None:
                        cb_final(state.fun_val)
                    z[0:nth] = [
                        z1 - z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])]
                    iter_num = state.iter_num
                    Jopt = state.fun_val

                if self.iprint > -1:
                    print(' L-BFGS-B done in %d iterations.' % iter_num)

            else:
                raise (Exception("\033[1mUnknown solver\033[0m"))

            t_solve = time.time() - t_solve
            return z[0:nth], Jopt, t_solve

        z, Jopt, t_solve1 = train_model('Adam', adam_epochs, z, np.inf)
        z, Jopt, t_solve2 = train_model('LBFGS', lbfgs_epochs, z, Jopt)
        t_solve = t_solve1+t_solve2

        # Zero coefficients smaller than zero_coeff in absolute value
        for i in range(len(z)):
            z[i] = np.array(z[i])
            z[i][np.abs(z[i]) <= self.zero_coeff] = 0.
        self.params = z

        # Check model sparsity
        sparsity = dict()
        sparsity["nonzero_parameters"] = [np.sum([np.sum(np.abs(z[i]) > self.zero_coeff) for i in range(
            len(z))]), np.sum([z[i].size for i in range(len(z))])]

        self.Jopt = Jopt
        self.t_solve = t_solve
        self.sparsity = sparsity
        return

    def parallel_fit(self, Y, U, init_fcn, seeds, n_jobs=None):
        """
        Fits the model in parallel using multiple seeds.

        Parameters:
            Y : ndarray
                Training dataset: output data. Y must be a N-by-ny numpy array.
            U : ndarray
                Training dataset: input data. U must be a N-by-nu numpy array.
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
            n_jobs = cpu_count()  # Use all available cores by default

        return Parallel(n_jobs=n_jobs)(delayed(single_fit)(seed) for seed in seeds)

    def sparsity_analysis(self):
        line = "-"*50 + "\n"
        txt = "Model sparsity:\n" + line
        txt += "%d nonzero model parameters out of %d (%6.2f%% sparsity)" % (
            self.sparsity["nonzero_parameters"][0], self.sparsity["nonzero_parameters"][1], 100*(1.-self.sparsity["nonzero_parameters"][0]/self.sparsity["nonzero_parameters"][1]))
        txt += "\n" + line
        return txt
