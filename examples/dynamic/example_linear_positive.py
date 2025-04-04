"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

System identification of a positive linear system.

(C) 2024 A. Bemporad, May 4, 2024
"""

import matplotlib.pyplot as plt
from jax_sysid.utils import compute_scores
from jax_sysid.models import LinearModel
import numpy as np

plotfigs = True  # set to True to plot figures

# Data generation
seed = 3  # for reproducibility of results
np.random.seed(seed)

nx = 4  # number of states
ny = 1  # number of outputs
nu = 1  # number of inputs

N_train = 1000  # number of training data
N_test = 1000  # number of test data
Ts = 1.  # sample time

# True linear dynamics (which is not a positive linear system)
At = np.random.rand(nx, nx); At[0,0] = -0.1; At[3,2] = -0.3; At[2,1] = -0.5
# makes matrix strictly Schur
At = At/np.max(np.abs(np.linalg.eig(At)[0]))*0.95
Bt = np.random.rand(nx, nu); Bt[0,0] = -0.1
Ct = np.random.rand(ny, nx); Ct[0,0] = -0.1
Dt = np.zeros((ny, nu))  # no direct feedthrough

qy = 0.1  # measurement noise std
qx = 0.0  # process noise std

U_train = np.random.randn(N_train, nu) # input excitation
x0_train = np.random.rand(nx) # nonegative initial state
# Create true model to generate the training dataset
truemodel = LinearModel(nx, ny, nu, feedthrough=False)
# truemodel.ss(feedthrough=False) # make model a linear state-space model without direct feedthrough
truemodel.init(params=[At, Bt, Ct], x0=x0_train)
Y_train, X_train = truemodel.predict(x0_train, U_train, qx, qy)
ymax = np.max(np.abs(Y_train))
umax = np.max(np.abs(U_train))
Ys_train = Y_train/ymax # scale outputs
Us_train = U_train/umax # scale inputs

U_test = np.random.randn(N_test, nu)
x0_test = np.random.rand(nx)
Y_test, X_test = truemodel.predict(x0_test, U_test, qx, qy)
Ys_test = Y_test/ymax # use same scaling as for training data
Us_test = U_test/umax

# Perform system identification
model = LinearModel(nx, ny, nu, feedthrough=False)  # create linear model
# MSE loss + L2-regularization on initial state and model coefficients
model.loss(rho_x0=1.e-3, rho_th=1.e-2)

# define lower bounds for the model coefficients to make them all nonnegative
params_min = [np.zeros_like(thi) for thi in model.params]

# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=0, lbfgs_epochs=1000, params_min=params_min, x0_min=np.zeros(nx))
model.fit(Ys_train, Us_train)
t0 = model.t_solve

print(f"Elapsed time: {t0} s")

A,B,C = model.params
print(f"Matrix A:\n\n {A}\n")
print(f"Matrix B:\n\n {B}\n")
print(f"Matrix C:\n\n {C}\n")
print(f"Initial state x0:\n\n {model.x0}\n")

Yshat_train, _ = model.predict(model.x0, Us_train)
Yhat_train = Yshat_train*ymax  # unscale outputs

x0_test = model.learn_x0(Us_test, Ys_test)  # use RTS Smoother to learn x0
Yshat_test, _ = model.predict(x0_test, Us_test)
Yhat_test = Yshat_test*ymax  # unscale outputs
R2, R2_test, msg = compute_scores(
    Y_train, Yhat_train, Y_test, Yhat_test, fit='R2')

print(msg)

if plotfigs:
    T_train = np.arange(N_train)*Ts
    T_test = np.arange(N_test)*Ts
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(T_train[0:99], Y_train[0:99, 0], label='measured')
    ax[0].plot(T_train[0:99], Yhat_train[0:99, 0], label='jax-sysid')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_title('Training data')
    ax[1].plot(T_test[0:99], Y_test[0:99, 0], label='measured')
    ax[1].plot(T_test[0:99], Yhat_test[0:99, 0], label='jax-sysid')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_title('Test data')
    plt.show()

