"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Linear and nonlinear system identification examples with additional emphasis on the DC gain of the system.
Steady-state data are used to fit the DC gain of the system. In the special case of linear systems, the DC gain can be fit instead to a given matrix gain.

(C) 2024 A. Bemporad, October 10, 2024
"""

import matplotlib.pyplot as plt
from jax_sysid.utils import standard_scale, unscale, compute_scores
from jax_sysid.models import Model, LinearModel
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:    
    jax.config.update("jax_enable_x64", True) # Enable 64-bit computations

runLTImodel =True
runNLmodel = True

plotfigs = True  # set to True to plot figures

# Data generation
seed = 3  # for reproducibility of results
np.random.seed(seed)

if runLTImodel:
    nx = 8  # number of states
    ny = 3  # number of outputs
    nu = 3  # number of inputs

    N_train = 1000  # number of training data
    N_test = 1000  # number of test data
    Ts = 1.  # sample time

    # True linear dynamics
    At = np.random.randn(nx, nx)
    # makes matrix strictly Schur
    At = At/np.max(np.abs(np.linalg.eig(At)[0]))*0.95
    Bt = np.random.randn(nx, nu)
    Ct = np.random.randn(ny, nx)
    Dt = np.zeros((ny, nu))  # no direct feedthrough

    DCgain_t=Ct@np.linalg.inv(np.eye(nx)-At)@Bt+Dt # true DC gain
    Ct = np.linalg.inv(DCgain_t)@Ct # scale C matrix to have unit DCgain_t
    DCgain_t = np.eye(ny)

    qy = 0.02  # output noise std
    qx = 0.02  # process noise std

    U_train = np.random.rand(N_train, nu)-0.5
    x0_train = np.random.randn(nx)
    # Create true model to generate the training dataset
    truemodel = LinearModel(nx, ny, nu, feedthrough=False)
    # truemodel.ss(feedthrough=False) # make model a linear state-space model without direct feedthrough
    truemodel.init(params=[At, Bt, Ct], x0=x0_train)
    Y_train, X_train = truemodel.predict(x0_train, U_train, qx, qy)
    Ys_train, ymean, ygain = standard_scale(Y_train)
    Us_train, umean, ugain = standard_scale(U_train)

    U_test = np.random.rand(N_test, nu)-0.5
    x0_test = np.random.randn(nx)
    Y_test, X_test = truemodel.predict(x0_test, U_test, qx, qy)
    Ys_test = (Y_test-ymean)*ygain  # use same scaling as for training data
    Us_test = (U_test-umean)*ugain

    model = LinearModel(nx, ny, nu, feedthrough=False)  # create linear model

    use_ss_data = True
    
    if use_ss_data:        
        # generate steady-state training data
        N_ss = 100
        Uss = 5*(np.random.rand(N_ss,nu)-.5) # steady-state input values
        Yss = np.array([DCgain_t@Uss[k] for k in range(N_ss)]) # steady-state output values
        Uss = (Uss-umean)*ugain
        Yss = (Yss-ymean)*ygain

        # Define loss to penalize deviations between predicted and measured steady-state values
        dcgain_loss = model.dcgain_loss(Uss, Yss, beta=1.)
    else:
        # scaled DC gain: Yss = DCgain_t*Uss
        #                 (Yss_scaled/ygain)+ymean = DCgain_t * (Uss_scaled/ugain+umean)#               #                  --> DCgain_scaled = ygain*DCgain_t/ugain
        DCgain = (DCgain_t.T*ygain).T/ugain
        dcgain_loss = model.dcgain_loss(DCgain = DCgain, beta=1.)

    # Define total loss = L2-regularization on initial state and model coefficients + DC-gain loss
    model.loss(rho_x0=1.e-3, rho_th=1.e-2, custom_regularization = dcgain_loss)

    # number of epochs for Adam and L-BFGS-B optimization
    model.optimization(adam_epochs=1000, lbfgs_epochs=1000)
    model.fit(Ys_train, Us_train)
    t0 = model.t_solve

    print(f"Elapsed time: {t0} s")
    Yshat_train, _ = model.predict(model.x0, Us_train)
    Yhat_train = unscale(Yshat_train, ymean, ygain)

    x0_test = model.learn_x0(Us_test, Ys_test)  # use RTS Smoother to learn x0
    Yshat_test, _ = model.predict(x0_test, Us_test)
    Yhat_test = unscale(Yshat_test, ymean, ygain)
    R2, R2_test, msg = compute_scores(
        Y_train, Yhat_train, Y_test, Yhat_test, fit='R2')

    print(msg)

    # Print eigenvalues of identified A matrix
    A, B, C, D = model.ssdata()
    DCgain_scaled=C@np.linalg.inv(np.eye(nx)-A)@B+D # DC gain of identified model on scaled data
    # scaled DC gain: (Yss-ymean)*ygain = DCgain_scaled*(Uss-umean)*ugain
    #                 --> DCgain = (DCgain_scaled*ugain)/ygain
    DCgain = (DCgain_scaled.T/ygain).T*ugain
    
    print(f"DC gain of true model:\n {DCgain_t}")
    print(f"DC gain of identified model:\n {DCgain}")
    
    if plotfigs:
        T_train = np.arange(N_train)*Ts
        T_test = np.arange(N_test)*Ts
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(T_train[0:99], Y_train[0:99, 0], label='measured')
        ax[0].plot(T_train[0:99], Yhat_train[0:99, 0], label='jax-sysid')
        ax[0].legend()
        ax[0].set_title('R2-score (training data)')
        ax[1].plot(T_test[0:99], Y_test[0:99, 0], label='measured')
        ax[1].plot(T_test[0:99], Yhat_test[0:99, 0], label='jax-sysid')
        ax[1].legend()
        ax[1].set_title('R2-score (test data)')
        plt.show()

if runNLmodel:
    np.random.seed(seed)
    
    nx = 3  # number of states
    ny = 1  # number of outputs
    nu = 1  # number of inputs

    N_train = 100  # number of training data
    N_test = 1000  # number of test data
    Ts = 1.  # sample time

    B = np.random.randn(nx, nu)
    C = np.random.randn(ny, nx)

    @jax.jit
    def truesystem(x0, U, D, qx, qy):    
        # system generating the training and test dataset (fast jax.lax.scan version)
        @jax.jit
        def true_dynamics(x, ud):
            u = ud[:nu]
            d = ud[nu:]
            xnext = jnp.array([.5*jnp.sin(x[0]) + B[0, :]@u * jnp.cos(x[1]/2.) + qx * d[0],
                    .6*jnp.sin(x[0]+x[2]) + B[1, :]@u * jnp.arctan(x[0]+x[1]) + qx * d[1],
                    .4*jnp.exp(-x[1]) + B[2, :]@u * jnp.sin(-x[0]/2.) + qx * d[2]])
            y = jnp.arctan(C @ x**3) + qy * d[3:]
            return xnext, y
        _, Y = jax.lax.scan(true_dynamics, jnp.array(x0), jnp.hstack((U,D)))
        return Y

    # Generate transient data
    qy = 0.01  # output noise std
    qx = 0.01  # process noise std
    U_train = np.random.rand(N_train, nu)-0.5
    D_train = np.random.randn(N_train, nx+ny)
    x0_train = np.zeros(nx)
    Y_train = truesystem(x0_train, U_train, D_train, qx, qy)

    U_test = np.random.rand(N_test, nu)-0.5
    D_test = np.random.randn(N_test, nx+ny)
    x0_test = np.zeros(nx)
    Y_test = truesystem(x0_test, U_test, D_test, qx, qy)

    # generate steady-state training and test data
    N_train_ss = 100
    N_test_ss = 100
    N = N_train_ss + N_test_ss
    Uss = 5*(np.random.rand(N)-.5) # steady-state input values
    M = 100 # number of data for each transient experiment
    D = 0.*np.random.randn(M,nx+ny)

    # simulate system for each steady-state input value and get average of last L samples
    print("Generating steady-state data ... ")
    L=10
    Yss = np.array([np.mean(truesystem(np.zeros(nx),Uss[k]*np.ones((M,1)),D,0.01,0.01)[-L:], axis=0).item() for k in range(N)])
    print("done.")

    # scale data
    Ys_train, ymean, ygain = standard_scale(Y_train)
    Us_train, umean, ugain = standard_scale(U_train)
    Ys_test = (Y_test-ymean)*ygain  # use same scaling as for training data
    Us_test = (U_test-umean)*ugain
    Uss = (Uss-umean)*ugain
    Yss = (Yss-ymean)*ygain

    # Define nonlinear model
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
    # initialize model coefficients
    model.init(params=[A, B, C, W1, W2, W3, b1, b2, W4, W5, b3, b4])

    # Define loss to penalize deviations between predicted and measured steady-state values
    dcgain_loss = model.dcgain_loss(Uss[:N_train_ss], Yss[:N_train_ss], beta=1.)
    
    # Define total loss = L2-regularization on initial state and model coefficients + DC-gain loss
    model.loss(rho_x0=1.e-4, rho_th=1.e-4, custom_regularization = dcgain_loss)
    
    # number of epochs for Adam and L-BFGS-B optimization
    model.optimization(adam_epochs=1000, lbfgs_epochs=2000)

    model.fit(Ys_train, Us_train)
    t0 = model.t_solve

    print(f"Elapsed time: {t0} s")
    Yshat_train, _ = model.predict(model.x0, Us_train)
    Yhat_train = unscale(Yshat_train, ymean, ygain)

    # use RTS Smoother to learn x0
    x0_test = model.learn_x0(Us_test, Ys_test, RTS_epochs=10)
    Yshat_test, _ = model.predict(x0_test, Us_test)
    Yhat_test = unscale(Yshat_test, ymean, ygain)
    R2, R2_test, msg = compute_scores(
        Y_train, Yhat_train, Y_test, Yhat_test, fit='R2')

    print(msg)
    # print(model.sparsity_analysis()) # only useful when tau_th>0

    if plotfigs:
        T_train = np.arange(N_train)*Ts
        T_test = np.arange(N_test)*Ts
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(T_train[0:99], Y_train[0:99, 0], label='measured')
        ax[0].plot(T_train[0:99], Yhat_train[0:99, 0], label='jax-sysid')
        ax[0].legend()
        ax[0].set_title('R2-score (training data)')
        ax[1].plot(T_test[0:99], Y_test[0:99, 0], label='measured')
        ax[1].plot(T_test[0:99], Yhat_test[0:99, 0], label='jax-sysid')
        ax[1].legend()
        ax[1].set_title('R2-score (test data)')
        plt.show()

    # Simulate model for each steady-state input value and get average of last L samples
    print("Generating steady-states on test inputs:")
    Yss_test = np.empty((N_test_ss,ny))
    for k in tqdm(range(N_test_ss)):
        y,_ = model.predict(np.zeros(nx),Uss[N_train_ss+k]*np.ones((M,1)))
        Yss_test[k,:] = y[-1]
        print(".", end="")
    print("done.")

    if plotfigs:
        plt.figure(figsize=(8,4))
        plt.scatter(Uss[N_train_ss:],Yss[N_train_ss:], label="measured")
        plt.scatter(Uss[N_train_ss:],Yss_test, label="predicted")
        plt.legend()
        plt.xlabel('steady-state input')
        plt.ylabel('steady-state output')
        plt.title('steady-state I/O data')
