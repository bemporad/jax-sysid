"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Linear system identification example with stability enforcement.

(C) 2024 A. Bemporad, June 11, 2024
"""

import matplotlib.pyplot as plt
from jax_sysid.utils import standard_scale, unscale, compute_scores, print_eigs
from jax_sysid.models import LinearModel
import numpy as np

plotfigs = True  # set to True to plot figures

# Data generation
seed = 2  # for reproducibility of results
np.random.seed(seed)

nx = 3  # number of states
ny = 1  # number of outputs
nu = 1  # number of inputs

N_train = 1000  # number of training data
N_test = 1000  # number of test data
Ts = 1.  # sample time

# True linear dynamics
At = np.array([[1.0001, 0.5, 0.5], [0., 0.9, -.2], [0., 0., 0.7]])
#At = np.array([[1.0001-.1, 0.5, 0.5], [0., 0.9, -.2], [0., 0., 0.7]]) # As. stable, but ||A||_2>1
Bt = np.round(np.random.randn(nx, nu)*10000.)/10000.
Ct = np.round(np.random.randn(ny, nx)*10000.)/10000.
Dt = np.zeros((ny, nu))  # no direct feedthrough

qy = 0.05  # output noise std
qx = 0.01  # process noise std


U_train = np.random.rand(N_train, nu)-0.5
x0_train = np.random.randn(nx)
# Create true model to generate the training dataset
truemodel = LinearModel(nx, ny, nu, feedthrough=False)
# truemodel.ss(feedthrough=False) # make model a linear state-space model without direct feedthrough
truemodel.init(params=[At, Bt, Ct], x0=x0_train)

Y_train, X_train = truemodel.predict(x0_train, U_train, qx, qy)
yscale=np.max(np.abs(Y_train))
Ys_train=Y_train/yscale

U_test = np.random.rand(N_test, nu)-0.5
x0_test = np.random.randn(nx)
Y_test, X_test = truemodel.predict(x0_test, U_test, qx, qy)

model = LinearModel(nx, ny, nu, feedthrough=False, sigma=0.5)  # create linear model

# Try enforcing stability of the identified linear model
model.force_stability(rho_A=1.e3, epsilon_A=1.e-3)
    
model.loss(rho_x0=1.e-3, rho_th=1.e-2)

# MSE loss + L2-regularization on initial state and model coefficients
model.loss(rho_x0=1.e-4, rho_th=1.e-4, xsat=1.e4)
# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=3000, lbfgs_epochs=5000)
model.fit(Ys_train, U_train)
t0 = model.t_solve
model.params[1]=model.params[1]*yscale

print(f"Elapsed time: {t0} s")

x0_train = model.learn_x0(U_train, Y_train, LBFGS_refinement=True)  # use RTS Smoother to learn x0

Yhat_train, _ = model.predict(x0_train, U_train)

x0_test = model.learn_x0(U_test, Y_test, LBFGS_refinement=True)  # use RTS Smoother to learn x0
Yhat_test, _ = model.predict(x0_test, U_test)
R2, R2_test, msg = compute_scores(
    Y_train, Yhat_train, Y_test, Yhat_test, fit='R2')

print(msg)

# Print eigenvalues of identified A matrix
A, B, C, D = model.ssdata()
print_eigs(A, num_digits=8)


# PyQt3D                5.15.6
# PyQt5                 5.15.9
# PyQt5-sip             12.12.1
# PyQt6                 6.5.1
# PyQt6-Qt6             6.5.1
# PyQt6-sip             13.5.1
# PyQtChart             5.15.6
# PyQtDataVisualization 5.15.5
# PyQtNetworkAuth       5.15.5
# PyQtPurchasing        5.15.5
# PyQtWebEngine         5.15.6


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
