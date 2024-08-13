"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Linear system identification example.

(C) 2024 A. Bemporad, February 25, 2024
"""

import matplotlib.pyplot as plt
from jax_sysid.utils import standard_scale, unscale, compute_scores, print_eigs
from jax_sysid.models import LinearModel
import numpy as np

plotfigs = False  # set to True to plot figures

# Data generation
seed = 3  # for reproducibility of results
np.random.seed(seed)

nx = 8  # number of states
ny = 3  # number of outputs
nu = 3  # number of inputs

N_train = 5000  # number of training data
N_test = 1000  # number of test data
Ts = 1.  # sample time

# True linear dynamics
At = np.random.randn(nx, nx)
# makes matrix strictly Schur
At = At/np.max(np.abs(np.linalg.eig(At)[0]))*0.95
Bt = np.random.randn(nx, nu)
Ct = np.random.randn(ny, nx)
Dt = np.zeros((ny, nu))  # no direct feedthrough

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

# Perform system identification
model = LinearModel(nx, ny, nu, feedthrough=False)  # create linear model
# MSE loss + L2-regularization on initial state and model coefficients
model.loss(rho_x0=1.e-3, rho_th=1.e-2)
# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=0, lbfgs_epochs=1000)
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
print_eigs(A)

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

########################################################################################
# Now add L1-regularization with penalty tau_th on the model coefficients
model.loss(rho_x0=1.e-3, rho_th=1.e-2, tau_th=0.03)
model.init()  # reinitialize model coefficients
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
print(model.sparsity_analysis())

########################################################################################
# Use group-Lasso regularization with penalty tau_g to reduce model order
model.loss(rho_x0=1.e-3, rho_th=1.e-2, tau_g=0.1)
model.group_lasso_x()  # introduce group-Lasso penalty to reduce the number of states
model.init()  # reinitialize model coefficients
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
print(model.sparsity_analysis())

########################################################################################
# Use group-Lasso regularization with penalty tau_g to reduce number of inputs
Bt[:, 2] = Bt[:, 2]/1000.  # make input #3 almost irrelevant
truemodel.init(params=[At, Bt, Ct], x0=x0_train)
Y_train, X_train = truemodel.predict(x0_train, U_train, qx, qy)
Ys_train, ymean, ygain = standard_scale(Y_train)
Us_train, umean, ugain = standard_scale(U_train)

U_test = np.random.rand(N_test, nu)-0.5
x0_test = np.random.randn(nx)
Y_test, X_test = truemodel.predict(x0_test, U_test, qx, qy)
Ys_test = (Y_test-ymean)*ygain  # use same scaling as for training data
Us_test = (U_test-umean)*ugain

model.loss(rho_x0=1.e-3, rho_th=1.e-2, tau_g=0.15)
model.group_lasso_u()  # introduce group-Lasso penalty to reduce the number of inputs
model.init()  # reinitialize model coefficients
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
print(model.sparsity_analysis())

########################################################################################
# Multiple experiments
model.loss(rho_x0=1.e-3, rho_th=1.e-2, tau_th=0.03)
model.init()  # reinitialize model coefficients
model.fit([Ys_train[0:int(N_train/3)], Ys_train[int(N_train/3):int(2*N_train/3)], Ys_train[int(2*N_train/3):]],
          [Us_train[0:int(N_train/3)], Us_train[int(N_train/3):int(2*N_train/3)], Us_train[int(2*N_train/3):]])
t0 = model.t_solve

print(f"Elapsed time: {t0} s")
# simulate the entire experiment
Yshat_train, _ = model.predict(model.x0[0], Us_train)
Yhat_train = unscale(Yshat_train, ymean, ygain)

x0_test = model.learn_x0(Us_test, Ys_test)  # use RTS Smoother to learn x0
Yshat_test, _ = model.predict(x0_test, Us_test)
Yhat_test = unscale(Yshat_test, ymean, ygain)
R2, R2_test, msg = compute_scores(
    Y_train, Yhat_train, Y_test, Yhat_test, fit='R2')

print(msg)
print(model.sparsity_analysis())
