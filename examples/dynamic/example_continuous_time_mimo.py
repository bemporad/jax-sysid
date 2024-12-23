"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Nonlinear system identification example in continuous time using a multi-input multi-output Wiener model

            dx(t)/dt = Ax(t) + Bu(t)
                y(t) = fy(Cx(t))

(C) 2024 A. Bemporad, December 23, 2024
"""

from jax_sysid.utils import standard_scale, compute_scores
from models import CTModel, LinearModel
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import diffrax

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

seed = 3  # for reproducibility of results
np.random.seed(seed)

nx = 3  # number of states
ny = 2  # number of outputs
nu = 2  # number of inputs

# Data generation - Generate data from a discrete-time model
N_train = 500  # number of training data
N_test = 500  # number of test data
Ts = 5.0  # sample time

At = np.diag(np.array([.98,.95,.92]))
Bt = np.random.randn(nx, nu)
Ct = np.random.randn(ny, nx)
Dt = np.zeros((ny, nu))  # no direct feedthrough
truemodel = LinearModel(nx, ny, nu, feedthrough=False)
# truemodel.ss(feedthrough=False) # make model a linear state-space model without direct feedthrough
truemodel.init(params=[At, Bt, Ct])
def nonlinearity(y):
    return y*(1.+.5*np.sin(y/2.))
x0=np.zeros(nx)
U_train = np.random.randn(N_train, nu)
Y_train = nonlinearity(truemodel.predict(x0, U_train)[0])
Y_train += 0.0*np.random.randn(N_train, ny)  # add noise
T_train = np.arange(0.,N_train*Ts,Ts)

U_test = np.random.randn(N_test, nu)
Y_test = nonlinearity(truemodel.predict(x0, U_test)[0])
Y_test += 0.0*np.random.randn(N_test, ny)  # add noise
T_test = np.arange(0.,N_test*Ts,Ts)

Ys_train, ymean, ygain = standard_scale(Y_train)
Us_train, umean, ugain = standard_scale(U_train)

Ys_test = (Y_test-ymean)*ygain
Us_test = (U_test-umean)*ugain

Ys_train = Ys_train.reshape(-1,ny)
Ys_test = Ys_test.reshape(-1,ny)

# Perform system identification
def activation(x):
    return x / (1. + jnp.exp(-x))
    #return jnp.maximum(0., x)

@jax.jit
def state_fcn(x, u, t, params):
    # This computes dx/dt = f(x,u) for continuous-time systems
    A, B, C1, C2, C3, b1, b2 = params
    return A@x+B@u

@jax.jit
def output_fcn(x, u, t, params):
    A, B, C1, C2, C3, b1, b2 = params
    return C1@activation(C2@x+b1)+C3@x+b2

nny = 20  # number of hidden neurons in output nonlinearity

def init_fcn(seed):
    np.random.seed(seed)
    A = -np.eye(nx) # start with a stable linear system
    B = np.random.randn(nx, nu)     
    C1 = .1*np.random.randn(ny, nny)
    C2 = .1*np.random.randn(nny, nx)
    C3 = np.random.randn(ny, nx)
    b1 = np.zeros(nny)
    b2 = np.zeros(ny)
    
    return [A, B, C1, C2, C3, b1, b2]


# initialize model coefficients
model = CTModel(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn)
model.integration_options(ode_solver=diffrax.Heun(), dt0=0.5)
model.init(params=init_fcn(seed=1))
# L2-regularization on initial state and model coefficients
model.loss(rho_x0=1.e-4, rho_th=1.e-4, train_x0=True) # default loss: integral of squared output prediction errors
# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=1000, lbfgs_epochs=3000)

# Fit model to training data
single_fit = False
if single_fit:
    model.fit(Ys_train, Us_train, T_train)
else:
    # Train from different initializations:
    models = model.parallel_fit(Ys_train, Us_train, T_train, init_fcn=init_fcn, seeds=range(10))
    R2 = model.find_best_model(models, Ys_train, Us_train, T_train)

x0_train=model.learn_x0(Us_train, Ys_train, T_train)
Yshat_train, Xhat_train = model.predict(x0_train, Us_train, T_train)

x0_test=model.learn_x0(Us_test, Ys_test, T_test)
T_test = np.arange(0.,T_test.size*Ts,Ts)
Yshat_test, _ = model.predict(x0_test, Us_test, T_test)

# Compute scores
R2_train, R2_test, msg = compute_scores(Ys_train, Yshat_train, Ys_test, Yshat_test)
print(msg)

# Plot results
fig, ax = plt.subplots(2, 2, figsize=(10, 6))
for k in range(2):
    str = 'training' if k==0 else 'test'    
    for j in range(2):
        ax[k,j].plot(T_train, Us_train, color=[.75,.75,.75], linestyle='--', linewidth=1., label='Input')
        ax[k,j].plot(T_train, Ys_train[:,j], label=fr'{str} - $y_{j+1}$')
        ax[k,j].plot(T_train, Yshat_train[:,j], label=fr'{str} - $\hat y_{j+1}$')
        ax[k,j].legend()
        ax[k,j].grid()
        ax[k,j].set_xlabel('Time')

