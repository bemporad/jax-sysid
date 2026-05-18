# -*- coding: utf-8 -*-
"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression/classification using JAX.

(C) 2024-2026 A. Bemporad
"""

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import diffrax
from joblib import Parallel, delayed, cpu_count
from jax_sysid.utils import lbfgs_options, compute_scores
from ._model import Model
from ._solvers import lbfgs_callback


class CTModel(Model):
    def __init__(self, nx, ny, nu, state_fcn, output_fcn=None, x0=None):
        """Create a nonlinear continuous-time model

            dx(t)/dt = fx(x(t),u(t),t)
                y(t) = fy(x(t),u(t),t)

        Parameters
        ----------
        nx : int
            Number of states.
        ny : int
            Number of outputs.
        nu : int
            Number of inputs.
        state_fcn : function
            State-update function defining dx/dt = state_fcn(x,u,t,params), where params are the trainable parameters of the model.
        output_fcn : function
            Output function defining y = output_fcn(x,u,t,params). If None, y=x[0:ny].
        x0 : ndarray or list of ndarrays or None
            Initial state vector or list of initial state vectors for multiple experiments. If None, x0=0.
        """

        super().__init__(nx, ny, nu, y_in_x=False, Ts=0.)

        self.state_fcn_ct = state_fcn
        self.state_fcn = None
        if output_fcn is None:
            @jax.jit
            def output_fcn(x, u, t, params):
                # if ny>nx, then the output is the first nx components of the state vector
                return x[0:self.ny]
        self.output_fcn_ct = output_fcn
        self.output_fcn = None
        self.params = None
        if x0 is not None:
            self.x0 = jnp.array(x0)
        else:
            self.x0 = jnp.zeros(self.nx)
        self.isLinear = False
        self.isInitialized = False
        return

    def integration_options(self, interpolation_type='zoh', ode_solver=diffrax.Heun(), dt0=None, max_steps=100000, stepsize_controller=None):
        """Set options for integrating the differential equations of the continuous-time model.

        Parameters
        ----------
        interpolation_type : str
            Interpolation type for the input data. Options are 'zoh' (zero-order hold) and 'linear'.
        ode_solver : diffrax.ODESolver
            ODE solver used for integrating the continuous-time model.
        dt0 : float
            Initial integration step size. If not provided, dt0 is set to (Tu[1]-Tu[0])/10.
        max_steps : int
            Maximum number of integration steps.
        stepsize_controller : diffrax.AbstractStepsizeController
            Controller for adaptive stepsize integration. If None, no controller is used.
            Example: stepsize_controller = diffrax.PIDController(rtol=1.e-3, atol=1.e-4, jump_ts=Tu)
        """

        if interpolation_type.lower() == 'zoh':
            iszoh=True
        elif interpolation_type.lower() == 'linear':
            iszoh=False
        else:
            raise ValueError(f"Invalid interpolation type {interpolation_type}")

        def input_fcn_gen(Tu, U):
            input_fcn=list()
            for k in range(self.nu):
                if iszoh:
                    # ZOH interpolation, with u(t) = u[k] for t in [Tu[k],Tu[k+1])
                    Tc, Uc = diffrax.rectilinear_interpolation(np.hstack((
                        -(Tu[1]-Tu[0]),Tu)), np.hstack((U[:,k],U[-1,k])))
                    input_fcn.append(diffrax.LinearInterpolation(ts=Tc, ys=Uc))
                else:
                    input_fcn.append(diffrax.LinearInterpolation(ts=Tu, ys=U[:,k]))
            return input_fcn

        self.input_fcn_gen = input_fcn_gen

        if dt0 is None:
            def dt0_fcn(Tu):
                return (Tu[1]-Tu[0])/10.
        else:
            def dt0_fcn(Tu):
                return dt0
        self.dt0_fcn = dt0_fcn

        self.ode_solver = ode_solver
        self.max_steps = max_steps
        self.stepsize_controller = stepsize_controller
        self.has_integration_options = True
        return

    def loss(self, output_loss=None, rho_x0=0.001, rho_th=0.01, tau_th=0.0, tau_g=0.0, group_lasso_fcn=None, zero_coeff=0., xsat=1000., train_x0=True, custom_regularization=None):
        """Define the overall loss function for system identification of continuous-time models.

        Parameters
        ----------
        output_loss : function
            Loss function loss=output_loss(Yhat,Y,T) penalizing the difference between the sequence
            Yhat of predicted and Y of measured outputs at sampling instants T = {t0,...,t(N-1)}.
            If None, the loss is defined as the integral of the squared error:

                       1     ⌠ t(N-1)
            loss = --------- ⎮   (yhat(t)-y(t))**2 dt
                   t(N-1)-t0 ⌡ t0

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
            def output_loss(Yhat, Y):
                # Use trapezoidal rule for integration
                T = self.Ty
                loss_t = ((Yhat.reshape(-1,self.ny)-Y.reshape(-1,self.ny))**2)
                return jnp.sum((loss_t[1:,:]+loss_t[0:-1,:])*jnp.diff(T).reshape(-1,1))/(T[-1]-T[0])/2.
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

    def prepare_fit_(self, U, T, Tu):
        """Prepare the model for fitting by setting up the input-output data and the ODE solver.
        """

        if not self.has_integration_options:
            raise ValueError(
                "Integration options not set. Use the integration_options method to set the ODE solver and other options.")

        if Tu is None:
            Tu = T
        self.Ty = T  # used to evaluate loss function

        input_fcn = self.input_fcn_gen(Tu, U)
        dt0 = self.dt0_fcn(Tu)

        @jax.jit
        def dxdt(t, x, params):
            u = jnp.array([input_fcn[k].evaluate(t) for k in range(self.nu)]).reshape(self.nu)
            return self.state_fcn_ct(x, u, t, params)

        @jax.jit
        def state_fcn(x, u, params):
            return x  # do nothing, just return the state
        self.state_fcn = state_fcn

        # evaluate input at output sample points
        Uy = jnp.array([input_fcn[k].evaluate(T) for k in range(self.nu)]).reshape(-1,self.nu)
        output_values = jax.jit(
            jax.vmap(self.output_fcn_ct, in_axes=[0, 0, 0, None]))

        @jax.jit
        def state_prediction(x, u, params):
            term = diffrax.ODETerm(dxdt)
            t0 = T[0]
            t1 = T[-1]
            saveat = diffrax.SaveAt(ts=jnp.array(T))
            if self.stepsize_controller is None:
                sol = diffrax.diffeqsolve(term, self.ode_solver, t0=t0, t1=t1, dt0=dt0,
                                          y0=x, args=params, saveat=saveat, max_steps=self.max_steps)
            else:
                sol = diffrax.diffeqsolve(term, self.ode_solver, t0=t0, t1=t1, dt0=dt0, y0=x, args=params,
                                          saveat=saveat, max_steps=self.max_steps, stepsize_controller=self.stepsize_controller)
            return sol.ys  # state sequence
        self.state_prediction = state_prediction

        @jax.jit
        def output_fcn(x, u, params):
            X = state_prediction(x, u, params)
            # return the entire (possibly multivariate) output sequence as one vector
            return output_values(X, Uy, T, params).reshape(-1)

        self.output_fcn = output_fcn
        return

    def fit(self, Y, U, T, Tu=None):
        """Fit the continuous-time model to input-output data.
        The model is trained by integrating the state-update function using an ODE solver from the diffrax library.

        Parameters
        ----------
        Y : ndarray
            Training dataset: output data. Y must be a N-by-ny numpy array.
        U : ndarray
            Training dataset: input data. U must be a N-by-nu numpy array.
        T: ndarray
            Time points corresponding to the output data.
        Tu: ndarray
            Time points corresponding to the input data. If None, Tu=T.
        """

        self.prepare_fit_(U=U, T=T, Tu=Tu)

        super().fit(Y.reshape(1, -1), jnp.zeros((1, self.nu)))  # pass dummy input data

        return

    def parallel_fit(
            self, Y, U, T, Tu=None, init_fcn=None, seeds=None, n_jobs=None):
        """
        Fits the continuous-time model in parallel using multiple seeds.

        The model is trained by integrating the state-update function over the time points of the dataset using an ODE solver from the diffrax library.

        Parameters
        ----------
        Y : ndarray
            Training dataset: output data. Y must be a N-by-ny numpy array.
        U : ndarray
            Training dataset: input data. U must be a N-by-nu numpy array.
        T: ndarray
            Time points corresponding to the output data.
        Tu: ndarray
            Time points corresponding to the input data. If None, Tu=T.
        init_fcn : function
            A function that initializes the model parameters given a seed.
        seeds : array
            The seeds used for initialization. If None, seeds = [0,1,...,n_jobs-1].
        n_jobs : int
            The number of parallel jobs to run (default is None, which means using all available cores).

        Returns:
        --------
            list: A list of fitted models.
        """

        self.prepare_fit_(U=U, T=T, Tu=Tu)

        if n_jobs is None:
            n_jobs = cpu_count()  # Use all available cores by default
        if seeds is None:
            seeds = range(n_jobs)

        def single_fit(seed):
            if not jax.config.jax_enable_x64:
                # Enable 64-bit computations
                jax.config.update("jax_enable_x64", True)
            self.init(params=init_fcn(seed))
            # if self.iprint > -1:
            #     print(
            #         "\033[1m" + f"Fitting model with seed = {seed} ... " + "\033[0m")

            Model.fit(self, Y.reshape(-1), jnp.zeros((1, self.nu))) # pass dummy input data

            # if self.iprint > -1:
            #     print("\033[1m" + f"Seed = {seed}: done." + "\033[0m")
            return self

        return Parallel(n_jobs=n_jobs)(delayed(single_fit)(seed) for seed in seeds)

    def predict(self, x0, U, T, Tu=None):
        """
        Predict the system's output and state trajectories y(t), x(t) at given time steps t listed in the array T,
        given the initial state x0 and input signal u(t) at given time steps t listed in the array Tu.

        Parameters
        ----------
        x0 : array-like
            Initial state of the system.
        U : ndarray
            Training dataset: input data. U must be a N-by-nu numpy array.
        T: ndarray
            Time points at which states and outputs are predicted.
        Tu: ndarray
            N-dimensional array with time points corresponding to the input data. If None, Tu=T.

        Returns:
        --------
        Y : array-like
            Predicted output sequence of the system at steps Ty.
        X : array-like
            Predicted state trajectory of the system at steps Ty.
        """
        if Tu is None:
            Tu = T

        self.prepare_fit_(U=U, T=T, Tu=Tu)

        X = self.state_prediction(x0, None, self.params)

        # evaluate input at output sample points
        input_fcn = self.input_fcn_gen(Tu, U)
        Uy = jnp.array([input_fcn[k].evaluate(T) for k in range(self.nu)]).reshape(-1,self.nu)

        output_values = jax.jit(
            jax.vmap(self.output_fcn_ct, in_axes=[0, 0, 0, None]))
        Y = output_values(X, Uy, T, self.params)
        return Y, X

    def learn_x0(self, U, Y, T, Tu=None, rho_x0=1.e-8, lbfgs_epochs=1000):
        """Estimate the initial state x0 by L-BFGS optimization.

        (C) 2024 A. Bemporad

        Parameters
        ----------
        U : ndarray
            Training dataset: input data. U must be a N-by-nu numpy array.
        Y : ndarray
            Training dataset: output data. Y must be a N-by-ny numpy array.
        T: ndarray
            Time points corresponding to the output data.
        Tu: ndarray
            Time points corresponding to the input data. If None, Tu=T.
        rho_x0 : float
            L2-regularization on initial state x0, 0.5*rho_x0*||x0||_2^2 (default: model.rho_x0)
        lbfgs_epochs : int
            Max number of L-BFGS iterations

        Returns
        -------
        array
            Optimal initial state x0.
        """

        if Tu is None:
            Tu = T

        self.prepare_fit_(U=U, T=T, Tu=Tu)
        self.Ty = T

        options = lbfgs_options(
            iprint=-1, iters=lbfgs_epochs, lbfgs_tol=1.e-6, memory=20)

        output_values = jax.jit(
            jax.vmap(self.output_fcn_ct, in_axes=[0, 0, 0, None]))

        # evaluate input at output sample points
        U = jnp.array([self.input_fcn_gen(Tu, U)[k].evaluate(T) for k in range(self.nu)]).reshape(-1,self.nu)

        @jax.jit
        def J(x0):
            X = self.state_prediction(x0, None, self.params)
            Yhat = output_values(X, U, T, self.params)
            loss = self.output_loss(Yhat, Y)
            loss += 0.5*rho_x0*jnp.sum(x0**2)
            return loss

        cb, cb_final = lbfgs_callback(options["maxfun"], options["iprint"], J)
        options.pop("iprint", None)
        options["disp"] = False
        solver = jaxopt.ScipyMinimize(
            fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options, callback=cb)
        x, state = solver.run(self.x0)
        if cb_final is not None:
            cb_final(state.fun_val)
        x = np.array(x)
        return x

    def find_best_model(self, models, Y, U, T, Tu=None, fit='R2', n_jobs=None, verbose=True):
        """
        Given a list of models, find the model that achieves the highest fit on a given dataset.

        Parameters
        ----------
        models : list
            List of models to evaluate.
        Y : ndarray
            Training dataset: output data. Y must be a N-by-ny numpy array.
        U : ndarray
            Training dataset: input data. U must be a N-by-nu numpy array.
        T: ndarray
            Time points corresponding to the output data.
        Tu: ndarray
            Time points corresponding to the input data. If None, Tu=T.
        fit : str or function, optional
            Metric to use for evaluating the fit. Default is 'R2', for other supported metrics
            (such as 'BFR', or 'RMSE', or 'Accuracy') see function utils/compute_scores().
            Alternatively, fit is a function fit(Y,Yhat) that takes two arguments: Y (=output data)
            and Yhat (=outputs predicted by each model) and returns a scalar fit value. The larger the
            value, the better the fit.
        n_jobs : int, optional
            Number of parallel jobs to run (default is None, which means using all available cores).
        verbose : bool
            If True, print scores for each model

        Returns
        -------
        score
            The score of the best model.

        The model object's parameters are updated with the parameters of the best model.
        """

        if not isinstance(models, list):
            raise Exception(
                "\033[1mPlease provide a list of models to compare.\033[0m")

        if Tu is None:
            Tu = T

        if isinstance(fit, str):
            def get_score(Y,Yhat):
                score, _, _ = compute_scores(Y, Yhat, fit=fit)
                if fit.lower() == 'rmse':
                    score = -score  # minimize RMSE
                return np.sum(score)/Y.shape[0] # average score over all outputs
        else:
            get_score = fit

        def single_score(k):
            x0 = models[k].learn_x0(U, Y, T, Tu)
            Yhat, _ = models[k].predict(x0, U, T, Tu)
            score = get_score(Y, Yhat)
            return score

        if len(models) == 1:
            self.params = models[0].params
            return single_score(0)  # if only one model, return its score directly

        if n_jobs is None:
            n_jobs = cpu_count()  # Use all available cores by default

        if verbose:
            print("Evaluating models...\n")

        scores = Parallel(n_jobs=n_jobs)(delayed(single_score)(k)
                                         for k in range(len(models)))

        best_id = np.argmax(np.sum(np.array(scores).reshape(len(models),-1),axis=1)) # get best score (best average score in case of multiple outputs)
        self.params = models[best_id].params

        if isinstance(fit, str):
            fit_name = fit
            if fit.lower() == 'rmse':
                scores = [-s for s in scores]
        else:
            fit_name = fit.__name__

        if verbose:
            print("Scores:")
            for k in range(len(models)):
                print(f"Model {k}: {fit_name} = {scores[k]}")
            print(f"Best model: {best_id}, score: {scores[best_id]}")
        return scores[best_id]
