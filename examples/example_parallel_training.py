"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Nonlinear system identification example using custom residual recurrent neural network model

            x(k+1) = A*x(k) + B*u(k) + fx(x(k),u(k))
              y(k) = C*x(k) + fy(x(k),u(k))

(C) 2024 A. Bemporad, August 13, 2024
"""

from jax_sysid.utils import standard_scale, unscale, compute_scores
from jax_sysid.models import Model, LinearModel, StaticModel, find_best_model
import jax
import jax.numpy as jnp
import numpy as np
from pmlb import fetch_data

runRNNModel = True
runLinearModel = False
runStaticModel = True

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

seed = 3  # for reproducibility of results
np.random.seed(seed)

if runRNNModel or runLinearModel:
    nx = 3  # number of states
    ny = 1  # number of outputs
    nu = 1  # number of inputs

    # Data generation
    N_train = 10000  # number of training data
    N_test = 1000  # number of test data
    Ts = 1.  # sample time

    B = np.random.randn(nx, nu)
    C = np.random.randn(ny, nx)

    def truesystem(x0, U, D, qx, qy):
        # system generating the training and test dataset
        N_train = U.shape[0]
        x = x0.copy()
        Y = np.empty((N_train, ny))
        X = np.empty((N_train, nx))
        for k in range(N_train):
            X[k] = x
            Y[k] = np.arctan(C @ x**3) + qy * D[k,nx:]
            x[0] = .5*np.sin(X[k,0]) + B[0, :]@U[k] * \
                np.cos(X[k,1]/2.) + qx * D[k,0]
            x[1] = .6*np.sin(X[k,0]+X[k,2]) + B[1, :]@U[k] * \
                np.arctan(X[k,0]+X[k,1]) + qx * D[k,1]
            x[2] = .4*np.exp(-X[k,1]) + B[2, :]@U[k] * \
                np.sin(-X[k,0]/2.) + qx * D[k,2]
        return Y, X

    qy = 0.01  # output noise std
    qx = 0.01  # process noise std
    U_train = np.random.rand(N_train, nu)-0.5
    D_train = np.random.randn(N_train, nx+ny)
    x0_train = np.zeros(nx)
    Y_train, _ = truesystem(x0_train, U_train, D_train, qx, qy)

    Ys_train, ymean, ygain = standard_scale(Y_train)
    Us_train, umean, ugain = standard_scale(U_train)

    U_test = np.random.rand(N_test, nu)-0.5
    D_test = np.random.randn(N_test, nx+ny)
    x0_test = np.zeros(nx)
    Y_test, _ = truesystem(x0_test, U_test, D_test, qx, qy)
    Ys_test = (Y_test-ymean)*ygain  # use same scaling as for training data
    Us_test = (U_test-umean)*ugain

if runRNNModel:
    # Perform system identification
    def sigmoid(x):
        return 1. / (1. + jnp.exp(-x))

    @jax.jit
    def state_fcn(x, u, params):
        A, B, C, W1, W2, W3, b1, b2, W4, W5, b3, b4 = params
        return A@x+B@u+W3@sigmoid(W1@x+W2@u+b1)+b2

    @jax.jit
    def output_fcn(x, u, params):
        A, B, C, W1, W2, W3, b1, b2, W4, W5, b3, b4 = params
        return C@x+W5@sigmoid(W4@x+b3)+b4


    model = Model(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn)

    nnx = 5  # number of hidden neurons in state-update function
    nny = 5  # number of hidden neurons in output function


    def init_fcn(seed):
        np.random.seed(seed)
        A = 0.5*np.eye(nx)
        B = 0.1*np.random.randn(nx, nu)
        C = 0.1*np.random.randn(ny, nx)
        W1 = 0.1*np.random.randn(nnx, nx)
        W2 = 0.5*np.random.randn(nnx, nu)
        W3 = 0.5*np.random.randn(nx, nnx)
        b1 = np.zeros(nnx)
        b2 = np.zeros(nx)
        W4 = 0.5*np.random.randn(nny, nx)
        W5 = 0.5*np.random.randn(ny, nny)
        b3 = np.zeros(nny)
        b4 = np.zeros(ny)
        return [A, B, C, W1, W2, W3, b1, b2, W4, W5, b3, b4]


    # initialize model coefficients
    model.init(params=init_fcn(seed=1))
    # L2-regularization on initial state and model coefficients
    model.loss(rho_x0=1.e-4, rho_th=1.e-4)
    # number of epochs for Adam and L-BFGS-B optimization
    model.optimization(adam_epochs=0, lbfgs_epochs=2000)

    models = model.parallel_fit(
        Ys_train, Us_train, init_fcn=init_fcn, seeds=range(10))

    # Find model that achieves best fit on test data
    best_model, best_R2 = find_best_model(models, Ys_test, Us_test, fit='R2')
    print(f"\nBest R2-score achieved on test data = {best_R2}")

if runLinearModel:
    # Parallel training of linear models. 
    # -----------------------------------
    model = LinearModel(nx, ny, nu, feedthrough=False)
    model.loss(rho_x0=1.e-3, rho_th=1.e-6)
    # number of epochs for Adam and L-BFGS-B optimization
    model.optimization(adam_epochs=0, lbfgs_epochs=1000)
    models = model.parallel_fit(Ys_train, Us_train, seeds=range(10))
    
    best_model, best_R2 = find_best_model(models, Ys_test, Us_test, fit='R2')
    print(f"\nBest R2-score achieved on test data = {best_R2}")

if runStaticModel:
    # Parallel training of static models:
    U, Y = fetch_data("529_pollen", return_X_y=True)
    tau_th = 0.002
    zero_coeff = 1.e-4
    Y = np.atleast_2d(Y).T

    # Data generation
    seed = 3  # for reproducibility of results
    np.random.seed(seed)

    ny = 1  # number of outputs
    N, nu = U.shape  # nu = number of inputs

    N_train = int(N*.75)  # number of training data
    N_test = N-N_train  # number of test data

    U_train = U[:N_train]
    Y_train = Y[:N_train]

    Ys_train, ymean, ygain = standard_scale(Y_train)
    Us_train, umean, ugain = standard_scale(U_train)

    U_test = U[N_train:]
    Y_test = Y[N_train:]
    Ys_test = (Y_test-ymean)*ygain  # use same scaling as for training data
    Us_test = (U_test-umean)*ugain


    @jax.jit
    def output_fcn(u, params):
        W1, b1, W2, b2 = params
        y = W1@u.T+b1
        y = W2@jnp.arctan(y)+b2
        return y.T


    model = StaticModel(ny, nu, output_fcn)
    nn = 10  # number of neurons


    def init_fcn(seed):
        np.random.seed(seed)
        W1 = np.random.randn(nn, nu)
        b1 = np.random.randn(nn, 1)
        W2 = np.random.randn(1, nn)
        b2 = np.random.randn(1, 1)
        return [W1, b1, W2, b2]


    # L1+L2-regularization on initial state and model coefficients
    model.loss(rho_th=1.e-4, tau_th=tau_th)
    # number of epochs for Adam and L-BFGS-B optimization
    model.optimization(adam_epochs=0, lbfgs_epochs=500)

    seeds = range(10)
    models = model.parallel_fit(Ys_train, Us_train, init_fcn=init_fcn, seeds=seeds)

    id = 0
    for model in models:
        Yshat_train = model.predict(Us_train)
        Yhat_train = unscale(Yshat_train, ymean, ygain)
        Yshat_test = model.predict(Us_test)
        Yhat_test = unscale(Yshat_test, ymean, ygain)
        R2, R2_test, msg = compute_scores(
            Y_train, Yhat_train, Y_test, Yhat_test, fit='R2')
        print(f"seed = {seeds[id]}: {msg}")
        id += 1

    print("Parallel training done.")