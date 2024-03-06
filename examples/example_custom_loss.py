"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

System identification of linear model with binary outputs.

(C) 2024 A. Bemporad, march 2, 2024
"""

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from jax_sysid.models import Model
from jax_sysid.utils import compute_scores

plotfigs = True  # set to True to plot figures

if plotfigs:
    plt.ion()
    plt.close('all')

# Data generation
seed = 3  # for reproducibility of results
np.random.seed(seed)

# nonlinear system with binary outputs (A. Bemporad, "Training recurrent neural networks by sequential least squares and the alternating direction method of multipliers," Automatica, vol. 156, pp. 111183, October 2023). 
A = np.array([[.8, .2, -.1], [0, .9, .1], [.1, -.1, .7]])
B = np.array([[-1.0], [.5], [1.0]])
C = np.array([[-2.0, 1.5, 0.5]])
N=2000
U=np.zeros((N,1))
Y=np.zeros((N,1))
nx=3
ny=1
nu=1
qy=0.05
qx=0.05
u = np.random.rand()
x=np.zeros((nx,1))
for i in range(N):
    U[i] = u
    Y[i] = float(C @ (x+x**3/3)  -4. + qy*np.random.randn()>=0)
    x = A @ x*(0.9+0.1*np.sin(x)) + B * u*(1-u**3) + qx*np.random.randn(nx,1)
    if np.random.rand() > 0.9:
        u = np.random.rand()
Ts=1.0 # sample time

N_train = 1000  # number of training data
N_test = N-N_train  # number of test data

U_train = U[0:N_train]
Y_train = Y[0:N_train]
U_test = U[N_train:]
Y_test = Y[N_train:]

# Perform system identification
jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations
@jax.jit
def state_fcn(x,u,params):
    W1,W2,W3,b1,b2,W4,W5,b3,b4=params
    return W3@jnp.tanh(W1@x+W2@u+b1)+b2    
def sigmoid(x):
    return 1. / (1. + jnp.exp(-x))  
@jax.jit
def output_fcn(x,u,params):
    W1,W2,W3,b1,b2,W4,W5,b3,b4=params
    return sigmoid(W5@jnp.tanh(W4@x+b3)+b4)

model = Model(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn)

nnx = 5 # number of hidden neurons in state-update function
nny = 5  # number of hidden neurons in output function

W1 = 0.1*np.random.randn(nnx,nx)
W2 = 0.5*np.random.randn(nnx,nu)
W3 = 0.5*np.random.randn(nx,nnx)
b1 = np.zeros(nnx)
b2 = np.zeros(nx)
W4 = 0.5*np.random.randn(nny,nx)
W5 = 0.5*np.random.randn(ny,nny)
b3 = np.zeros(nny)
b4 = np.zeros(ny)
model.init(params=[W1,W2,W3,b1,b2,W4,W5,b3,b4]) # initialize model coefficients

epsil=1.e-4
@jax.jit
def cross_entropy_loss(Yhat,Y):
    loss=jnp.sum(-Y*jnp.log(epsil+Yhat)-(1.-Y)*jnp.log(epsil+1.-Yhat))/Y.shape[0]
    return loss
model.loss(rho_x0=0.01, rho_th=0.001, output_loss=cross_entropy_loss) 
model.optimization(adam_epochs=2000, lbfgs_epochs=2000) # number of epochs for Adam and L-BFGS-B optimization
model.fit(Y_train, U_train)
t0 = model.t_solve

print(f"Elapsed time: {t0} s")
Yhat_train, _ = model.predict(model.x0, U_train)
Yhat_train = (Yhat_train>=0.5).astype(float)

N0 = N_test  # number of data used to learn x0
x0_test = model.learn_x0(U_test[0:N0], Y_test[0:N0], RTS_epochs=1, LBFGS_refinement=True)  # use RTS Smoother to learn x0
Yhat_test, _ = model.predict(x0_test, U_test)
Yhat_test = (Yhat_test>=0.5).astype(float)

# Compute accuracy scores
acc_train, acc_test, msg = compute_scores(Y_train, Yhat_train, Y_test, Yhat_test, fit='Accuracy')
print(msg)

if plotfigs:
    T_train = np.arange(N_train)*Ts
    T_test = np.arange(N_test)*Ts
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(T_train, Y_train, label='measured')
    ax[0].plot(T_train, Yhat_train, label='jax-sysid')
    ax[0].legend()
    ax[0].set_title('R2-score (training data)')
    ax[1].plot(T_test, Y_test, label='measured')
    ax[1].plot(T_test, Yhat_test, label='jax-sysid')
    ax[1].legend()
    ax[1].set_title('R2-score (test data)')
    plt.show()
