"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

The examples included here show the use of bound constraints on some of the model parameters
to fit a static convex function to data.

(C) 2024 A. Bemporad, May 2, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from jax_sysid.utils import compute_scores
from jax_sysid.models import StaticModel
import jax
import flax.linen as nn

seed = 4 # for reproducibility of results
np.random.seed(seed)

plotfigs = True  # set to True to plot figures

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

for example in ['1d','2d']:
    print(f'Running {example}-example')

    # Data generation
    if example == '1d':
        U = np.random.rand(5000, 1)
        Y = -np.log(U).reshape(-1) # convex function to approximate
        ymax = np.max(Y)
        Y /= ymax # normalize outputs for better numerical conditioning
        Y = np.atleast_2d(Y).T

    elif example == '2d':
        U1, U2 = np.meshgrid(np.linspace(-2.,2.,100),np.linspace(-2.,2.,100))
        Y = np.sin((U1**2+U2**2)/5.) + np.exp(-((U1-1.)**2+(U2-1.)**2)) # function to approximate
        
        U=np.hstack((U1.reshape(-1,1),U2.reshape(-1,1)))
        Y=Y.reshape(-1,1)

    tau_th = 0. # L1-regularization term
    zero_coeff = 1.e-4 # small coefficients are set to zero when L1-regularization is used

    ny = Y.shape[1]  # number of outputs
    N, nu = U.shape  # nu = number of inputs

    # input convex function model [Amos, Xu, Kolter, 2017]
    act = nn.elu # activation function, must be convex and non decreasing on the domain of interest
    @jax.jit
    def output_fcn(x,params):
        W0y,W1z,W1y,W2z,W2y,b0,b1,b2=params
        z1 = act(W0y @ x.T + b0)
        z2 = act(W1z @ z1 + W1y @ x.T + b1)
        y = W2z @ z2 + W2y @ x.T + b2
        return y.T

    model = StaticModel(ny, nu, output_fcn)
    n1,n2 = 10,10  # number of neurons
    params=[np.random.rand(n1, nu), #W0y
            np.random.rand(n2, n1), #W1z
            np.random.rand(n2, nu), #W1y
            np.random.rand(1, n2), #W2z
            np.random.randn(1, nu), #W2y
            np.random.randn(n1, 1), #b0
            np.random.randn(n2, 1), #b1
            np.random.randn(1, 1)] #b2
    model.init(params=params)
    model.loss(rho_th=1.e-8, tau_th=tau_th)

    params_min = [np.zeros((n1,nu)), np.zeros((n2,n1)), np.zeros((n2,nu)), np.zeros((1,n2)),  
                -np.inf*np.ones((1,nu)), -np.inf*np.ones((n1,1)), -np.inf*np.ones((n2,1)), -np.inf*np.ones((1,1))] # some weights are nonnegative
    #params_min = None
    params_max = None # no upper bounds
    model.optimization(adam_epochs=1000, lbfgs_epochs=1000, params_min=params_min, params_max=params_max)
    model.fit(Y,U)
    t0 = model.t_solve

    print(f"Elapsed time: {t0} s")
    Yhat = model.predict(U.reshape(-1,nu))
    R2, _, msg = compute_scores(Y, Yhat, None, None, fit='R2')

    print(msg)
    if tau_th > 0:
        print(model.sparsity_analysis())

    if plotfigs:
        if example == '1d':
            # T = np.arange(N)
            # plt.figure(figsize=(6, 4))
            # plt.plot(T[0:99], Y[0:99, 0], label='measured')
            # plt.plot(T[0:99], Yhat[0:99, 0], label='jax-sysid')
            # plt.legend()
            # plt.grid()
            # plt.title('training data')    
            # plt.show()

            W0y,W1z,W1y,W2z,W2y,b0,b1,b2=model.params
            @jax.jit
            def f(x):
                z1 = act(W0y*x + b0)
                z2 = act(W1z @ z1 + W1y*x + b1)
                y = W2z@z2 + W2y*x + b2
                return y*ymax

            U=np.linspace(0.1,1,100)
            Y = -np.log(U)
            Yhat=np.array([f(u) for u in U]).reshape(-1)
            fig, ax = plt.subplots(3, 1, figsize=(4, 8))
            ax[0].plot(U,Y,label='true function')
            ax[0].plot(U,Yhat,label='approximation')
            ax[0].legend()
            ax[0].grid()
            
            df = jax.jacfwd(f)
            dY = -1./U
            dYhat=np.array([df(u) for u in U]).squeeze()
            ax[1].plot(U,dY,label='1st derivative - true')
            ax[1].plot(U,dYhat,label='1st derivative - approx')
            ax[1].legend()
            ax[1].grid()
            
            d2f = jax.jacfwd(jax.jacfwd(f))
            d2Y = 1./U**2
            d2Yhat=np.array([d2f(u) for u in U]).squeeze()
            ax[2].plot(U,d2Y,label='2nd derivative - true')
            ax[2].plot(U,d2Yhat,label='2nd derivative - approx')
            ax[2].legend()
            ax[2].grid()

            if np.all(d2Yhat>=0.):
                print('Approximation appears to be convex')    
            plt.show()
        elif example == '2d':
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].contour(U1,U2,Y.reshape(U1.shape))
            ax[0].grid()
            ax[0].set_title('True function')
            ax[1].contour(U1,U2,Yhat.reshape(U1.shape))
            ax[1].grid()
            ax[1].set_title('Convex approximation')
            plt.show()
            
