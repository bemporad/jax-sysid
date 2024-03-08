"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Nonlinear system identification example, using recurrent neural network based on flax.linen library.

(C) 2024 A. Bemporad, March 6, 2024
"""
from flax import linen as nn
import matplotlib.pyplot as plt
from jax_sysid.utils import standard_scale, unscale, compute_scores
from jax_sysid.models import RNN
import jax
import numpy as np

plotfigs = True  # set to True to plot figures

if plotfigs:
    plt.ion()
    plt.close('all')

# Data generation
seed = 3  # for reproducibility of results
np.random.seed(seed)

nx = 3  # number of states
ny = 1  # number of outputs
nu = 1  # number of inputs

N_train = 10000  # number of training data
N_test = 1000  # number of test data
Ts = 1.  # sample time

B = np.random.randn(nx, nu)
C = np.random.randn(ny, nx)


def truesystem(x0, U, qx, qy):
    # system generating the training and test dataset
    N_train = U.shape[0]
    x = x0.copy()
    Y = np.empty((N_train, ny))
    for k in range(N_train):
        if k == 0:
            x = x0
        else:
            x[0] = .5*np.sin(x[0]) + B[0, :]@U[k-1] * \
                np.cos(x[1]/2.) + qx * np.random.randn(1)
            x[1] = .6*np.sin(x[0]+x[2]) + B[1, :]@U[k-1] * \
                np.arctan(x[0]+x[1]) + qx * np.random.randn(1)
            x[2] = .4*np.exp(-x[1]) + B[2, :]@U[k-1] * \
                np.sin(-x[0]/2.) + qx * np.random.randn(1)
        Y[k] = np.arctan(C @ x**3) + qy * np.random.randn(ny)
    return Y


qy = 0.01  # output noise std
qx = 0.01  # process noise std
U_train = np.random.rand(N_train, nu)-0.5
x0_train = np.zeros(nx)
Y_train = truesystem(x0_train, U_train, qx, qy)

Ys_train, ymean, ygain = standard_scale(Y_train)
Us_train, umean, ugain = standard_scale(U_train)

U_test = np.random.rand(N_test, nu)-0.5
x0_test = np.zeros(nx)
Y_test = truesystem(x0_test, U_test, qx, qy)
Ys_test = (Y_test-ymean)*ygain  # use same scaling as for training data
Us_test = (U_test-umean)*ugain

# Perform system identification
jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

# state-update function


class FX(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=5)(x)
        x = nn.swish(x)
        x = nn.Dense(features=5)(x)
        x = nn.swish(x)
        x = nn.Dense(features=nx)(x)
        return x

# output function


class FY(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=5)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=ny)(x)
        return x


model = RNN(nx, ny, nu, FX=FX, FY=FY, x_scaling=0.1)
# L1+L2-regularization on initial state and model coefficients
model.loss(rho_x0=1.e-4, rho_th=1.e-4, tau_th=0.0002)
# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=0, lbfgs_epochs=2000)
model.fit(Ys_train, Us_train)
t0 = model.t_solve

print(f"Elapsed time: {t0} s")
Yshat_train, _ = model.predict(model.x0, Us_train)
Yhat_train = unscale(Yshat_train, ymean, ygain)

# use RTS Smoother to learn x0
x0_test = model.learn_x0(Us_test, Ys_test, RTS_epochs=3)
Yshat_test, _ = model.predict(x0_test, Us_test)
Yhat_test = unscale(Yshat_test, ymean, ygain)
R2, R2_test, msg = compute_scores(
    Y_train, Yhat_train, Y_test, Yhat_test, fit='R2')

print(msg)
print(model.sparsity_analysis())

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
