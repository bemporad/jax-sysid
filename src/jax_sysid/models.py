# -*- coding: utf-8 -*-
"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression/classification using JAX.

(C) 2024 A. Bemporad, March 6, 2024
"""

import numpy as np
import time
import jax
import jax.numpy as jnp
import jaxopt
from functools import partial
from scipy.linalg import solve
import tqdm
import sys
from jax_sysid.utils import lbfgs_options, vec_reshape

epsil_lasso = 1.e-16  # tolerance used in groupLassoReg functions to prevent 0/0 = nan in the Jacobian vector when the argument is the zero vector
default_small_tau_th = 1.e-8  # add some small L1-regularization.
# This is parameter epsilon in Lemma 2 of the paper
# A. Bemporad, "Linear and nonlinear system identification under L1- and group-Lasso regularization via L-BFGS-B", submitted for publication and available on arXiv, 2024.


@jax.jit
def l2reg(z):
    """
    Compute the L2 regularization cost for a given set of parameters.

    Parameters
    ----------
    z : list
        List of parameters.

    Returns
    -------
    float
        The L2 regularization cost.
    """
    cost = 0.
    for w in z:
        cost += jnp.sum(w ** 2)
    return cost


@jax.jit
def l1reg(z):
    """
    Compute the L1 regularization cost for a given set of parameters.

    Parameters
    ----------
    z : list
        List of parameters.

    Returns
    -------
    float
        The L1 regularization cost.
    """
    cost = 0.
    for w in z:
        cost += jnp.sum(jnp.abs(w))
    return cost


@jax.jit
def linreg(z):
    """
    Compute the linear cost given by the sum of all parameters.

    Parameters
    ----------
    z : list
        List of parameters.

    Returns
    -------
    float
        The linear cost.
    """
    cost = 0.
    for w in z:
        cost += jnp.sum(w)
    return cost


def optimization_base(object, adam_eta=0.001, adam_epochs=0, lbfgs_epochs=1000, iprint=50, memory=10, lbfgs_tol=1.e-16):
    """Define the optimization parameters for the system identification problem.

    Parameters
    ----------
    object : object
        Model object to be trained by optimization.
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
        Tolerance for L-BFGS-B iterations.
    Returns
    -------
    object  
        The model object modified with the set of optimization parameters.
    """
    object.adam_eta = adam_eta
    object.adam_epochs = adam_epochs
    object.lbfgs_epochs = lbfgs_epochs
    object.iprint = iprint
    object.memory = memory
    object.lbfgs_tol = lbfgs_tol
    return object


def adam_solver(JdJ, z, solver_iters, adam_eta, iprint, params_min=None, params_max=None):
    """
    Solves a nonlinear optimization problem using the Adam optimization algorithm.

    Parameters
    ----------
    JdJ : callable
        A function that computes the objective function value and its gradient.
    z : list
        A list of numpy arrays representing the initial guess of the optimization variables.
    solver_iters : int
        The number of iterations for the solver.
    adam_eta : float
        The learning rate for the Adam algorithm.
    iprint : int
        Verbosity level. Set to -1 to disable printing.
    params_min : list, optional
        A list of numpy arrays representing the lower bounds of the optimization variables z.
        Lower bounds are enforced by clipping the variables during the iterations.
    params_max : list, optional
        A list of numpy arrays representing the upper bounds of the optimization variables z.
        Upper bounds are enforced by clipping the variables during the iterations.

    Returns
    -------
    tuple
        A tuple containing the optimized variables `z` and the objective function value `Jopt`.
    """
    if iprint > -1:
        iters_tqdm = tqdm.tqdm(total=solver_iters, desc='Iterations', ncols=30,
                               bar_format='{percentage:3.0f}%|{bar}|', leave=True, position=0)
        loss_log = tqdm.tqdm(
            total=0, position=1, bar_format='{desc}')
        nvars = sum([zi.size for zi in z])
        print("Solving NLP with Adam (%d optimization variables) ..." % nvars)

    nz = len(z)
    fbest = np.Inf
    v = [np.zeros(zi.shape) for zi in z]
    m = [np.zeros(zi.shape) for zi in z]
    beta1 = 0.9
    beta2 = 0.99
    beta1t = beta1
    beta2t = beta2
    epsil = 1e-8

    ismin = (params_min is not None)
    ismax = (params_max is not None)
    isbounded = ismin or ismax
    if isbounded:
        # Clip the initial guess, when required
        for j in range(nz):
            if ismin:
                z[j] = np.maximum(z[j], params_min[j])
            if ismax:
                z[j] = np.minimum(z[j], params_max[j])

    for k in range(solver_iters):
        f, df = JdJ(z)
        if fbest > f:
            fbest = f
            zbest = z.copy()
        for j in range(nz):
            m[j] = beta1*m[j] + (1 - beta1) * df[j]
            v[j] = beta2*v[j] + (1 - beta2) * df[j]**2
            # classical Adam step
            z[j] -= adam_eta/(1 - beta1t)*m[j] / \
                (np.sqrt(v[j]/(1 - beta2t)) + epsil)
        if isbounded:
            for j in range(nz):
                if ismin:
                    z[j] = np.maximum(z[j], params_min[j])
                if ismax:
                    z[j] = np.minimum(z[j], params_max[j])

        beta1t *= beta1
        beta2t *= beta2

        if iprint > -1:
            str = f"    f = {f: 10.6f}, f* = {fbest: 8.6f}"
            ndf = np.sum([np.linalg.norm(df[i])
                          for i in range(nz)])
            str += f", |grad f| = {ndf: 8.6f}"
            str += f", iter = {k}"
            loss_log.set_description_str(str)
            iters_tqdm.update(1)

    z = zbest.copy()
    Jopt, _ = JdJ(z)  # = fbest

    if iprint > -1:
        iters_tqdm.close()
        loss_log.close()

    return z, Jopt


@jax.jit
def xsat(x, sat):
    """
    Apply saturation to the state value.

    Parameters
    ----------
    x : jax.numpy.ndarray
        The state value.
    sat : float
        The saturation limit.

    Returns
    -------
    jax.numpy.ndarray
        The saturated value of x.
    """
    return jnp.minimum(jnp.maximum(x, -sat), sat)  # hard saturation


def get_bounds(z, epsil_lasso, params_min, params_max):
    """
    Utility function to create bounds for L-BFGS-B when splitting positive and negative parts   

    Parameters
    ----------
    z : list
        List of parameters.
    epsil_lasso : float
        Small positive value used to constrain the positive and negative parts to be strictly positive.
    params_min : list of arrays
        List of the same structure as self.params of lower bounds for the parameters.
    params_max : list of arrays
        List of the same structure as self.params of upper bounds for the parameters.
    """

    isbounded = (params_min is not None) or (params_max is not None)
    nz = len(z)

    lb = list()
    ub = list()
    if not isbounded:
        for h in range(2):
            for i in range(nz):
                lb.append(np.zeros_like(z[i])+epsil_lasso)
                ub.append(np.inf*np.ones_like(z[i]))
    else:
        # We have bounds on the parameters:
        #     lb <= x  <= ub with y,z>=epsil_lasso and x = y-z
        #
        # Then, we impose:
        #       max(0,lb) +epsil_lasso <= y <= max(0,ub) +epsil_lasso
        #       max(0,-ub)+epsil_lasso <= z <= max(0,-lb)+epsil_lasso
        #
        # Note: we could eliminate some variables when the lower and upper bounds
        # are both equal to epsil_lasso
        for i in range(nz):
            # positive part y
            zi = jnp.zeros_like(z[i])
            lb.append(jnp.maximum(params_min[i], zi)+epsil_lasso)
            ub.append(jnp.maximum(params_max[i], zi)+epsil_lasso)
        for i in range(nz):
            # negative part z
            zi = jnp.zeros_like(z[i])
            lb.append(jnp.maximum(-params_max[i], zi)+epsil_lasso)
            ub.append(jnp.maximum(-params_min[i], zi)+epsil_lasso)
    return (lb, ub)


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
        self.isLinear = False  # By default, the model is nonlinear
        self.Ts = Ts  # sample time

        self.output_loss = None
        self.rho_x0 = None
        self.rho_th = None
        self.tau_th = None
        self.tau_g = None
        self.zero_coeff = None
        self.xsat = None
        self.train_x0 = None

        self.adam_eta = None
        self.adam_epochs = None
        self.lbfgs_epochs = None
        self.iprint = None
        self.memory = None

        self.isInitialized = False

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
            If None, use standard mean squared error loss=sum((Yhat-Y)^2)/Y.shape[0]
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
            Custom regularization term, a function of the model parameters and initial state
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
        self.group_lasso_fcn = group_lasso_fcn
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
            tau_th = default_small_tau_th  # add some small L1-regularization, see Lemma 2

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
                    self.x0_min = [self.x0_min]*Nexp # repeat the same initial-state bound on all experiments
                if len(self.x0_min) is not Nexp:
                    self.x0_min = [self.x0_min[0]]*Nexp # number of experiments has changed, repeat the same initial-state bound on all experiments
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
            Train a linear state-space model using the specified solver.

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
                    z[i] = jnp.maximum(zi, 0.)+epsil_lasso
                    z[nth+i] = -jnp.minimum(zi, 0.)+epsil_lasso

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
                    lb = self.params_min
                    ub = self.params_max
                    if self.train_x0:
                        lb.append(self.x0_min)
                        ub.append(self.x0_max)
                        
                z, Jopt = adam_solver(
                    JdJ, z, solver_iters, self.adam_eta, self.iprint, lb, ub)

            elif solver == "LBFGS":
                # L-BFGS-B params (no L1 regularization)
                options = lbfgs_options(
                    min(self.iprint, 90), solver_iters, self.lbfgs_tol, self.memory)

                if self.iprint > -1:
                    print(
                        "Solving NLP with L-BFGS (%d optimization variables) ..." % nvars)

                if isGroupLasso or isL1reg:
                    bounds = get_bounds(
                        z[0:nth], epsil_lasso, self.params_min, self.params_max)
                    if self.train_x0:
                        bounds[0].append(self.x0_min)
                        bounds[1].append(self.x0_max)

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
                        if not self.isbounded:
                            solver = jaxopt.ScipyMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                            z, state = solver.run(z)
                        else:
                            lb = self.params_min
                            ub = self.params_max
                            if self.train_x0:
                                lb.append(self.x0_min)
                                ub.append(self.x0_max)
                            solver = jaxopt.ScipyBoundedMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                            z, state = solver.run(z, bounds=(lb, ub))
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

                        solver = jaxopt.ScipyBoundedMinimize(
                            fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                        z, state = solver.run(z, bounds=bounds)
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

                    solver = jaxopt.ScipyBoundedMinimize(
                        fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                    z, state = solver.run(z, bounds=bounds)
                    z[0:nth] = [
                        z1 - z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])]
                    iter_num = state.iter_num
                    Jopt = state.fun_val

                print('L-BFGS-B done in %d iterations.' % iter_num)

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
            _, Xt = self.predict(x0[i], U[i], 0, 0)
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
            Process noise covariance matrix, by default 1.e-8*I
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
            if self.y_in_x:
                A, _ = self.params
            else:
                if self.feedthrough:
                    A, _, C, _ = self.params
                else:
                    A, _, C = self.params
        else:
            @jax.jit
            def Ck(x, u):
                return jax.jacrev(self.output_fcn)(x, u=u, params=self.params)

            @jax.jit
            def Ak(x, u):
                return jax.jacrev(self.state_fcn)(x, u=u, params=self.params)
            AA = np.empty((N, nx, nx))  # A(k)

        if rho_x0 is None:
            rho_x0 = self.rho_x0
        PP1 = np.empty((N, nx, nx))  # P(k | k)
        PP2 = np.empty((N, nx, nx))  # P(k + 1 | k)
        XX1 = np.empty((N, nx))  # x(k | k)
        XX2 = np.empty((N, nx))  # x(k + 1 | k)
        if R is None:
            R = np.eye(ny)
        if Q is None:
            Q = 1.e-8 * np.eye(nx)

        # Forward EKF pass:
        # L2-regularization on initial state x0, 0.5*rho_x0*||x0||_2^2
        P = np.eye(nx) / (rho_x0 * N)
        x = np.zeros(nx)

        if verbosity:
            therange = tqdm.trange(
                N, ncols=30, bar_format='{percentage:3.0f}%|{bar}|', leave=True)
        else:
            therange = range(N)

        for epoch in range(RTS_epochs):
            mse_loss = 0.0
            for k in therange:
                u = U[k]

                # measurement update
                y = self.output_fcn(x, u, self.params)
                if not isLinear:
                    C = Ck(x, u)
                PC = P @ C.T
                # M = PC / (R + C @PC) # this solves the linear system M*(R + C @PC) = PC
                # Note: Matlab's mrdivide A / B = (B'\A')' = np.linalg.solve(B.conj().T, A.conj().T).conj().T
                M = solve((R+C@PC), PC.T, assume_a='pos').T
                e = Y[k]-y
                mse_loss += np.sum(e**2)  # just for monitoring purposes
                x += M@e  # x(k | k)
                P -= M@PC.T  # P(k|k)
                P = (P + P.T)/2.
                PP1[k] = P
                XX1[k] = x

                # Time update
                if not isLinear:
                    A = Ak(x, u)
                    AA[k] = A
                P = A@P@A.T+Q
                # P = (P+P.T)/2.
                x = self.state_fcn(x, u, self.params)
                PP2[k] = P
                XX2[k] = x

            # RTS smoother pass:
            x = XX1[N-1]
            P = PP1[N-1]
            for k in range(N-1, -1, -1):
                if not isLinear:
                    A = AA[k]
                # G=(PP1[k]@AA[k].T)/PP2[k]
                G = solve(PP2[k], (PP1[k]@A.T).T, assume_a='pos').T
                x = XX1[k]+G@(x-XX2[k])
                P = PP1[k]+G@(P-PP2[k])@G.T

            if verbosity:
                sys.stdout.write('\033[F')
                print(
                    f"\nRTS smoothing, epoch: {epoch+1: 3d}/{RTS_epochs: 3d}, MSE loss = {mse_loss/N: 8.6f}")

        isstatebounded = self.x0_min is not None or self.x0_max is not None
        if isstatebounded:
            lb = self.x0_min
            if lb is None:
                lb = -np.inf*np.ones(nx)
            ub = self.x0_max
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
            if not isstatebounded:
                solver = jaxopt.ScipyMinimize(
                    fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options)
                x, state = solver.run(x)
            else:
                solver = jaxopt.ScipyBoundedMinimize(
                fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options)
                x, state = solver.run(x, bounds=(lb, ub))
                
            if verbosity:
                mse_loss = state.fun_val-.5*LBFGS_rho_x0*np.sum(x**2)
                print(
                    f"\nFinal loss MSE (after LBFGS refinement) = {mse_loss: 8.6f}")

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


class LinearModel(Model):
    def __init__(self, nx, ny, nu, feedthrough=False, y_in_x=False, x0=None, sigma=0.5, seed=0, Ts=None, ss=None):
        """Create the linear state-space model structure

            x(k+1) = A*x(k) + B*u(k)
            y(k) = C*x(k) + D*u(k)

        Parameters
        ----------
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

        if self.y_in_x:
            @jax.jit
            def state_fcn(x, u, params):
                A, B = params
                return A @ x + B @ u

            @jax.jit
            def output_fcn(x, u, params):
                return x[0:self.ny]
        else:
            if not feedthrough:
                @jax.jit
                def state_fcn(x, u, params):
                    A, B, _ = params
                    return A @ x + B @ u

                @jax.jit
                def output_fcn(x, u, params):
                    _, _, C = params
                    return C @ x
            else:
                @jax.jit
                def state_fcn(x, u, params):
                    A, B, _, _ = params
                    return A @ x + B @ u

                @jax.jit
                def output_fcn(x, u, params):
                    _, _, C, D = params
                    return C @ x + D @ u

        self.state_fcn = state_fcn
        self.output_fcn = output_fcn
        return

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

    def params2ABCD(self):
        A, B = self.params[0:2]
        if self.y_in_x:
            C = np.eye(self.ny)
            D = np.zeros((self.ny, self.nu))
        else:
            C = self.params[2]
            if self.feedthrough:
                D = self.params[3]
            else:
                D = np.zeros((self.ny, self.nu))
        return A, B, C, D

    def group_lasso_x(self):
        """Group-Lasso regularization on linear state-space matrices A,B,C and x0 to penalize the model order (number of states)
        """
        @jax.jit
        def groupLassoRegX(th, x0):
            cost = 0.
            A, B = th[0:2]
            if self.y_in_x:
                A, B = th[0:2]
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


class RNN(Model):
    def __init__(self, nx, ny, nu, FX, FY=None, y_in_x=False, x_scaling=0.1, y_scaling=1.0, x0=None, seed=0, Ts=None):
        """Create a recurrent neural network model structure

            x(k+1) = fx(x(k),u(k))
            y(k) = fy(x(k),u(k))

        Parameters
        ----------
        FX : subclass of flax.linen.nn.Module
            Feedforward neural-network function of the state update
        FY : subclass of flax.linen.nn.Module
            Feedforward neural-network function of the output
        y_in_x : bool
            If True, the output is included in the first ny components of the state vector and FY is not used
        x_scaling : float
            Scaling factor for the initial weight matrices of the state update function
        y_scaling : float
            Scaling factor for the weight matrices of the output function
        x0 : ndarray or list of ndarrays or None
            Initial state vector or list of initial state vectors for multiple experiments. If None, x0=0.
        seed : int
            Random seed for initializing the parameters of the neural networks (default: 0)
        Ts : float
            Sampling time
        """

        super().__init__(nx, ny, nu, y_in_x=y_in_x, Ts=Ts)

        key1, key2 = jax.random.split(jax.random.PRNGKey(seed), 2)

        fx = FX()
        # initialize parameters by passing a template vector
        thx = fx.init(key1, jnp.ones(nx+nu))['params']
        thx_flat, thx_tree = jax.tree_util.tree_flatten(thx)
        params = thx_flat
        self.nthx = len(thx_flat)
        self.thx_tree = thx_tree

        @jax.jit
        def state_fcn(x, u, params):
            thx = jax.tree_util.tree_unflatten(
                self.thx_tree, params[0:self.nthx])
            x = fx.apply({'params': thx}, jnp.hstack((x, u)))  # time update
            return x

        if not y_in_x:
            fy = FY()
            thy = fy.init(key2, jnp.ones(nx+nu))['params']
            thy_flat, thy_tree = jax.tree_util.tree_flatten(thy)
            params.extend(thy_flat)
            self.thy_tree = thy_tree

            @jax.jit
            def output_fcn(x, u, params):
                thy = jax.tree_util.tree_unflatten(
                    self.thy_tree, params[self.nthx:])
                y = fy.apply({'params': thy}, jnp.hstack(
                    (x, u)))  # predicted output
                return y
        else:
            @jax.jit
            def output_fcn(x, u, params):
                return x[0:self.ny]

        for i in range(len(params)):
            if i < self.nthx:
                params[i] *= x_scaling
            else:
                params[i] *= y_scaling

        self.state_fcn = state_fcn
        self.output_fcn = output_fcn
        self.params = params
        if x0 is not None:
            if isinstance(x0, list):
                Nexp = len(x0)
                self.x0 = [jnp.array(x0[i]) for i in range(Nexp)]
            else:
                self.x0 = jnp.array(x0)
        else:
            self.x0 = jnp.zeros(self.nx)
        self.isLinear = False
        self.isInitialized = True
        return


class StaticModel(object):
    """
    Base class of trainable static input/output models
    """

    def __init__(self, ny, nu, output_fcn=None):
        """
        Initialize model structure

        ny: number of outputs

        nu: number of inputs

        output_fcn: function handle to the output function y(k)=output_fcn(u(k),params).
        The function must have the signature f(u, params) where u is the input and params is a list of ndarrays of parameters. The function must be vectorized with respect to u, i.e., it must be able to handle a matrix of inputs u, one row per input value. The function must return a matrix of outputs y, where each row corresponds to an input in u.
        """
        self.ny = ny  # number of outputs
        self.nu = nu  # number of inputs
        self.output_fcn = output_fcn

        self.output_loss = None
        self.rho_x0 = None
        self.rho_th = None
        self.tau_th = None
        self.tau_g = None
        self.zero_coeff = None
        self.xsat = None
        self.train_x0 = None

        self.adam_eta = None
        self.adam_epochs = None
        self.lbfgs_epochs = None
        self.iprint = None
        self.memory = None

        self.isInitialized = False

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
            If None, use standard mean squared error loss=sum((Yhat-Y)^2)/Y.shape[0]
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
            Custom regularization term, a function of the model parameters 
            custom_regularization(params, x0).            
        """
        if output_loss is None:
            def output_loss(Yhat, Y): return jnp.sum((Yhat - Y)**2)/Y.shape[0]
        self.output_loss = output_loss
        self.rho_th = rho_th
        self.tau_th = tau_th
        self.tau_g = tau_g
        self.zero_coeff = zero_coeff
        self.group_lasso_fcn = group_lasso_fcn
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
            tau_th = default_small_tau_th

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
                    z[i] = jnp.maximum(zi, 0)+epsil_lasso
                    z[nth+i] = -jnp.minimum(zi, 0)+epsil_lasso

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

                if self.iprint > -1:
                    print(
                        "Solving NLP with L-BFGS (%d optimization variables) ..." % nvars)

                if not isGroupLasso:
                    if not isL1reg:
                        def J(z):
                            cost = loss(z) + self.rho_th*l2reg(z)
                            if isCustomReg:
                                cost += self.custom_regularization(z)
                            return cost

                        if (self.params_min is None) and (self.params_max is None):
                            solver = jaxopt.ScipyMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                            z, state = solver.run(z)
                        else:
                            solver = jaxopt.ScipyBoundedMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                            z, state = solver.run(z, bounds=(
                                self.params_min, self.params_max))

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

                        solver = jaxopt.ScipyBoundedMinimize(
                            fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)

                        bounds = get_bounds(
                            z[0:nth], epsil_lasso, self.params_min, self.params_max)
                        z, state = solver.run(z, bounds=bounds)
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

                    solver = jaxopt.ScipyBoundedMinimize(
                        fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                    bounds = get_bounds(
                        z[0:nth], epsil_lasso, self.params_min, self.params_max)
                    z, state = solver.run(z, bounds=bounds)
                    z[0:nth] = [
                        z1 - z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])]
                    iter_num = state.iter_num
                    Jopt = state.fun_val

                if self.iprint > -1:
                    print('L-BFGS-B done in %d iterations.' % iter_num)

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

    def sparsity_analysis(self):
        line = "-"*50 + "\n"
        txt = "Model sparsity:\n" + line
        txt += "%d nonzero model parameters out of %d (%6.2f%% sparsity)" % (
            self.sparsity["nonzero_parameters"][0], self.sparsity["nonzero_parameters"][1], 100*(1.-self.sparsity["nonzero_parameters"][0]/self.sparsity["nonzero_parameters"][1]))
        txt += "\n" + line
        return txt


class FNN(StaticModel):
    def __init__(self, ny, nu, FY, seed=0):
        """Create a feedforward neural network 

            y(k) = f(x(k))

        Parameters
        ----------
        ny : int
            Number of outputs
        nu : int    
            Number of inputs
        FY : subclass of flax.linen.nn.Module
            Feedforward neural-network function of the output
        seed : int
            Random seed for initializing the parameters of the neural network (default: 0)
        """

        key = jax.random.PRNGKey(seed)

        # initialize parameters by passing a template vector
        fy = FY()
        thy = fy.init(key, jnp.ones(nu))['params']
        thy_flat, thy_tree = jax.tree_util.tree_flatten(thy)
        params = thy_flat
        self.thy_tree = thy_tree

        @jax.jit
        def output_fcn(u, params):
            thy = jax.tree_util.tree_unflatten(self.thy_tree, params)
            y = fy.apply({'params': thy}, u)  # predicted output
            return y

        super().__init__(ny, nu, output_fcn=output_fcn)

        self.params = params
        self.isInitialized = True
        return
