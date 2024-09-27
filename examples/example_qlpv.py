"""
Quasi-LPV identification example.

(C) 2024 A. Bemporad, August 13, 2024
"""

from jax_sysid.utils import unscale, compute_scores, standard_scale
from jax_sysid.models import qLPVModel
import numpy as np
import jax
import flax.linen as nn
import matplotlib.pyplot as plt
import time

plotfigs = True  # set to True to plot figures

t0 = time.time()

# Data generation
seed = 0  # for reproducibility of results
np.random.seed(seed)

nx = 3  # number of states
ny = 1  # number of outputs
nu = 1  # number of inputs
# number of scheduling parameters (npar>=0. If npar=0, the model is linear time-invariant)
npar = 2
nn1, nn2 = 6, 6  # number of neurons in FNN defining the scheduling vector

N_train = 5000  # number of training data
N_test = 1000  # number of test data
Ts = 1.  # sample time

B = np.floor(np.random.randn(nx, nu)*10.)/10.
C = np.floor(np.random.randn(ny, nx)*10.)/10.


def truesystem(x0, U, qx, qy):
    # system generating the training and test dataset
    N_train = U.shape[0]
    x = x0.copy()
    Y = np.empty((N_train, ny))
    X = np.empty((N_train, nx))
    for k in range(N_train):
        X[k] = x
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
    return Y, X


qy = 0.01  # output noise std
qx = 0.01  # process noise std
U_train = np.random.rand(N_train, nu)-0.5
x0_train = np.zeros(nx)
Y_train, _ = truesystem(x0_train, U_train, qx, qy)

Ys_train, ymean, ygain = standard_scale(Y_train)
Us_train, umean, ugain = standard_scale(U_train)

U_test = np.random.rand(N_test, nu)-0.5
x0_test = np.zeros(nx)
Y_test, _ = truesystem(x0_test, U_test, qx, qy)
Ys_test = (Y_test-ymean)*ygain  # use same scaling as for training data
Us_test = (U_test-umean)*ugain

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

rho_th = 1.e-4  # L2-regularization parameter for model coefficients
rho_x0 = 1.e-4  # L2-regularization parameter for initial state
adam_epochs = 1000
lbfgs_epochs = 5000
memory = 20  # memory parameter for L-BFGS-B optimization
iprint = 50  # -1  # verbosity level for L-BFGS-B optimization
train_x0 = True

# Initialize parameters of scheduling function (feedforward neural network)
W1x = np.random.randn(nn1, nx)
W1u = np.random.randn(nn1, nu)
b1 = np.zeros(nn1)
W2 = np.random.randn(nn2, nn1)
b2 = np.zeros(nn2)
W3 = np.random.randn(npar, nn2)
b3 = np.zeros(npar)

qlpv_params_init = [W1x, W1u, b1, W2, b2, W3, b3]

# Scheduling function
@jax.jit
def qlpv_fcn(x, u, qlpv_params):
    W1x, W1u, b1, W2, b2, W3, b3 = qlpv_params
    p = nn.sigmoid(
        W3 @ nn.swish(W2 @ nn.swish(W1x @ x + W1u @ u + b1) + b2) + b3)
    return p

# Define qLPV model
model = qLPVModel(nx, ny, nu, npar, qlpv_fcn, qlpv_params_init,
                feedthrough=False, y_in_x=False, x0=None, sigma=0.5, seed=0, Ts=None)
model.loss(rho_th=rho_th)
model.optimization(adam_epochs=adam_epochs,
                lbfgs_epochs=lbfgs_epochs, memory=memory, iprint=iprint)
model.fit(Ys_train, Us_train, LTI_training=True)

# Extract matrices defining qLPV structure
Alin, Blin, Clin, Dlin, Ap, Bp, Cp, Dp = model.ssdata()
# Extract parameters defining the scheduling function
qlpv_params = model.params[:model.nqlpv_params]

# Compute BFR scores
Yshat_train, X_train = model.predict(model.x0, Us_train)
Yhat_train = unscale(Yshat_train, ymean, ygain)
x0_test = model.learn_x0(
    Us_test, Ys_test, LBFGS_refinement=True, LBFGS_rho_x0=rho_x0, verbosity=0)
Yshat_test, X_test = model.predict(x0_test, Us_test)
Yhat_test = unscale(Yshat_test, ymean, ygain)

BFR_train, BFR_test, msg = compute_scores(
    Y_train, Yhat_train, Y_test, Yhat_test, fit='BFR')

if plotfigs:
    plt.figure(figsize=(8, 4))
    plt.plot(Y_test, label='True')
    plt.plot(Yhat_test, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('output (test data)')
    plt.legend()
    plt.show()

    p_test = np.array([qlpv_fcn(x, u, qlpv_params)
                    for x, u in zip(X_test, Us_test)])

    plt.figure(figsize=(8, 4))
    plt.plot(p_test)
    plt.grid()
    plt.title('scheduling vector')

# Repeat with group-Lasso regularization to possibly reduce the number of scheduling variables
model = qLPVModel(nx, ny, nu, npar, qlpv_fcn, qlpv_params_init,
                feedthrough=False, y_in_x=False, x0=None, sigma=0.5, seed=0, Ts=None)
model.loss(rho_th=rho_th, tau_th=0., tau_g=0.02, zero_coeff=1.e-4)
model.group_lasso_p()
model.optimization(adam_epochs=adam_epochs,
                lbfgs_epochs=lbfgs_epochs, memory=memory, iprint=iprint)
model.fit(Ys_train, Us_train, LTI_training=True)
Alin, Blin, Clin, Dlin, Ap, Bp, Cp, Dp = model.ssdata()
# parameters defining the scheduling function
qlpv_params = model.params[:model.nqlpv_params]

print(f"Ap = {Ap}\n\nBp = {Bp}\n\nCp = {Cp}\n\nDp = {Dp}")
removable_parameters = model.sparsity["removable_parameters"]

# Compute BFR scores
Yshat_train, X_train = model.predict(model.x0, Us_train)
Yhat_train = unscale(Yshat_train, ymean, ygain)
x0_test = model.learn_x0(
    Us_test, Ys_test, LBFGS_refinement=True, LBFGS_rho_x0=rho_x0, verbosity=0)
Yshat_test, X_test = model.predict(x0_test, Us_test)
Yhat_test = unscale(Yshat_test, ymean, ygain)

BFR_train2, BFR_test2, msg2 = compute_scores(
    Y_train, Yhat_train, Y_test, Yhat_test, fit='BFR')

if plotfigs:
    plt.figure(figsize=(8, 4))
    plt.plot(Y_test, label='True')
    plt.plot(Yhat_test, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('output (test data)')
    plt.legend()
    plt.show()

    p_test = np.array([qlpv_fcn(x, u, qlpv_params)
                    for x, u in zip(X_test, Us_test)])

    plt.figure(figsize=(8, 4))
    plt.plot(p_test)
    plt.grid()
    plt.title('scheduling vector')

# Now retrain the model with the reduced number of scheduling variables, using parallel training
new_npar = 1
def qlpv_param_init_fcn(seed):
    np.random.seed(seed)
    W1x = np.random.randn(nn1, nx)
    W1u = np.random.randn(nn1, nu)
    b1 = np.zeros(nn1)
    W2 = np.random.randn(nn2, nn1)
    b2 = np.zeros(nn2)
    W3 = np.random.randn(new_npar, nn2)
    b3 = np.zeros(new_npar)
    return [W1x, W1u, b1, W2, b2, W3, b3]

model = qLPVModel(nx, ny, nu, new_npar, qlpv_fcn, qlpv_params_init,
                  feedthrough=False, y_in_x=False, x0=None, sigma=0.5, seed=0, Ts=None)
model.loss(rho_th=rho_th)
model.optimization(adam_epochs=adam_epochs,
                   lbfgs_epochs=lbfgs_epochs, memory=memory, iprint=iprint)
models = model.parallel_fit(Ys_train, Us_train, qlpv_param_init_fcn=qlpv_param_init_fcn, seeds=range(10), n_jobs=10)
# Find model that achieves best fit on test data (this operation could be parallelized too)
best_R2=-np.inf
best_id = -1
msg3 = ''
id=0
for model in models:
    x0_test = model.learn_x0(Us_test, Ys_test)
    Yshat_train, _ = model.predict(model.x0, Us_train)
    Yhat_train = unscale(Yshat_train, ymean, ygain)
    Yshat_test, _ = model.predict(x0_test, Us_test)
    Yhat_test = unscale(Yshat_test, ymean, ygain)
    R2, R2_test, msg_id = compute_scores(
        Y_train, Yhat_train, Y_test, Yhat_test, fit='BFR')
    print(msg_id)
    if float(R2_test)>best_R2:
        best_R2 = float(R2_test)
        best_id = id
        msg3 = msg_id
    id+=1
best_model = models[best_id]

print(f"\nTraining results\n{'-'*30}")
print(f"#scheduling vars = {npar},               {msg}")
print(f"#scheduling vars = {npar} (group-Lasso), {msg2}")
for i in removable_parameters:
    print(f"  scheduling parameter #{i+1} was redundant and removed")
print(f"#scheduling vars = {new_npar},               {msg3}\n")

print(f"Elapsed time: {time.time()-t0} s")
