"""
Quasi-LPV identification example.

(C) 2024 A. Bemporad, August 13, 2024
"""

from jax_sysid.utils import unscale, compute_scores, standard_scale
from jax_sysid.models import qLPVModel, find_best_model
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
D_test = np.random.randn(N_train, nx+ny)
x0_test = np.zeros(nx)
Y_test, _ = truesystem(x0_test, U_test, D_test, qx, qy)
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
model.loss(rho_th=rho_th, rho_x0=rho_x0)
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
model_group = qLPVModel(nx, ny, nu, npar, qlpv_fcn, qlpv_params_init,
                feedthrough=False, y_in_x=False, x0=None, sigma=0.5, seed=0, Ts=None)
model_group.loss(rho_th=rho_th, rho_x0=rho_x0, tau_th=0., tau_g=0.07, zero_coeff=1.e-4)
model_group.group_lasso_p()
model_group.optimization(adam_epochs=adam_epochs,
                lbfgs_epochs=lbfgs_epochs, memory=memory, iprint=iprint)
model_group.fit(Ys_train, Us_train, LTI_training=True)
Alin, Blin, Clin, Dlin, Ap, Bp, Cp, Dp = model_group.ssdata()
# parameters defining the scheduling function
qlpv_params = model_group.params[:model.nqlpv_params]

print(f"Ap = {Ap}\n\nBp = {Bp}\n\nCp = {Cp}\n\nDp = {Dp}")
removable_parameters = model_group.sparsity["removable_parameters"]
kept_parameters = np.delete(np.arange(npar),removable_parameters)

# Compute BFR scores
Yshat_train, X_train = model_group.predict(model.x0, Us_train)
Yhat_train = unscale(Yshat_train, ymean, ygain)
x0_test = model_group.learn_x0(
    Us_test, Ys_test, LBFGS_refinement=True, LBFGS_rho_x0=rho_x0, verbosity=0)
Yshat_test, X_test = model_group.predict(x0_test, Us_test)
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
    plt.plot(p_test[:,kept_parameters])
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

model_single = qLPVModel(nx, ny, nu, new_npar, qlpv_fcn, qlpv_params_init,
                  feedthrough=False, y_in_x=False, x0=None, sigma=0.5, seed=0, Ts=None)
model_single.loss(rho_th=rho_th, rho_x0=rho_x0)
model_single.optimization(adam_epochs=adam_epochs,
                   lbfgs_epochs=lbfgs_epochs, memory=memory, iprint=iprint)
models_single = model_single.parallel_fit(Ys_train, Us_train, qlpv_param_init_fcn=qlpv_param_init_fcn, seeds=range(10))

# Find model that achieves best fit on test data
model_single, best_R2 = find_best_model(models_single, Ys_test, Us_test, fit='R2')
x0_test = model_single.learn_x0(Us_test, Ys_test)
Yshat_train, _ = model_single.predict(model_single.x0, Us_train)
Yhat_train = unscale(Yshat_train, ymean, ygain)
Yshat_test, _ = model_single.predict(x0_test, Us_test)
Yhat_test = unscale(Yshat_test, ymean, ygain)
R2, R2_test, msg3 = compute_scores(
        Y_train, Yhat_train, Y_test, Yhat_test, fit='BFR')

print(f"\nTraining results\n{'-'*30}")
print(f"#scheduling vars = {npar},               {msg}")
print(f"#scheduling vars = {npar} (group-Lasso), {msg2}")
for i in removable_parameters:
    print(f"  scheduling parameter #{i+1} was redundant and removed")
print(f"#scheduling vars = {new_npar},               {msg3}\n")

print(f"Elapsed time: {time.time()-t0} s")
