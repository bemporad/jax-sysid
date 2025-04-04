"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Nonlinear system identification example, using recurrent neural network based on flax.linen library.

(C) 2024 A. Bemporad, March 6, 2024
"""
from flax import linen as nn
import matplotlib.pyplot as plt
from jax_sysid.utils import standard_scale, unscale, compute_scores
from jax_sysid.models import RNN, find_best_model
import jax
import numpy as np

plotfigs = True  # set to True to plot figures

# Data generation
seed = 3  # for reproducibility of results
np.random.seed(seed)

nx = 3  # number of states
ny = 1  # number of outputs
nu = 1  # number of inputs

N_train = 1000  # number of training data
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
model.loss(rho_x0=1.e-4, rho_th=0., tau_th=4.e-4)
# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=0, lbfgs_epochs=2000)

train_single_model=True
if train_single_model:
    model.fit(Ys_train, Us_train)
    t0 = model.t_solve
    print(f"Elapsed time: {t0} s")
else:
    models=model.parallel_fit(Ys_train, Us_train, init_fcn=model.init_fcn, seeds=range(10))
    model, best_R2 = find_best_model(models, Ys_test, Us_test, fit='R2')

Yshat_train, _ = model.predict(model.x0, Us_train)
Yhat_train = unscale(Yshat_train, ymean, ygain)

# use RTS Smoother to learn x0
x0_test = model.learn_x0(Us_test, Ys_test, RTS_epochs=10)
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
    ax[0].set_title('Training data')
    ax[1].plot(T_test[0:99], Y_test[0:99, 0], label='measured')
    ax[1].plot(T_test[0:99], Yhat_test[0:99, 0], label='jax-sysid')
    ax[1].legend()
    ax[1].set_title('Test data')
    plt.show()
