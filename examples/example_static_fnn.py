"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Static feedforward neural network training based on flax.linen library.

(C) 2024 A. Bemporad, March 1, 2024
"""
from flax import linen as nn
from pmlb import fetch_data
import matplotlib.pyplot as plt
from jax_sysid.utils import standard_scale, unscale, compute_scores
from jax_sysid.models import FNN
import jax
import numpy as np

plotfigs = True  # set to True to plot figures

U, Y = fetch_data("529_pollen", return_X_y=True)
tau_th = 0.002
zero_coeff = 1.e-4
# U,Y = fetch_data("225_puma8NH", return_X_y=True); tau_th=0.001; zero_coeff=1.e-4
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

# Perform system identification
jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

# output function


class FY(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=20)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=20)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=ny)(x)
        return x


model = FNN(ny, nu, FY)
# L1+L2-regularization on initial state and model coefficients
model.loss(rho_th=1.e-4, tau_th=tau_th)
# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=0, lbfgs_epochs=500)
model.fit(Ys_train, Us_train)
t0 = model.t_solve

print(f"Elapsed time: {t0} s")
Yshat_train = model.predict(Us_train)
Yhat_train = unscale(Yshat_train, ymean, ygain)

Yshat_test = model.predict(Us_test)
Yhat_test = unscale(Yshat_test, ymean, ygain)
R2, R2_test, msg = compute_scores(
    Y_train, Yhat_train, Y_test, Yhat_test, fit='R2')

print(msg)
print(model.sparsity_analysis())

if plotfigs:
    T_train = np.arange(N_train)
    T_test = np.arange(N_test)
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
