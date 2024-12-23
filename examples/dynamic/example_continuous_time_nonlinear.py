"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Nonlinear system identification example in continuous time using a recurrent neural network model

            dx(t)/dt = fx(x(t),u(t),t)
                y(t) = fy(x(t),u(t),t)

(C) 2024 A. Bemporad, December 22, 2024
"""

from jax_sysid.utils import standard_scale, compute_scores
from models import CTModel
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

nx = 2  # number of states
ny = 1  # number of outputs
nu = 1  # number of inputs

# Data generation
Ts = 0.5  # sample time of control input (time units)
Tstop = 150 # total excitation time (time units)
T_train = np.arange(0.,Tstop,Ts)
#U_train = np.sin(0.002*T_train**2)  # input signal
u0=4.*np.random.rand()-2.
U_train = np.empty(T_train.shape).reshape(-1,1)
for i in range(T_train.size):
    if np.random.randn()>0.95:
        u0=4.*np.random.rand()-2.
    U_train[i] = u0
x0_train = np.array([0.,0.])  # initial state

Ts=0.5
Tstop = 150 # total excitation time (time units)
T_test = np.arange(0.,Tstop,Ts)
U_test = (np.cos(0.1*T_test)+.5*(T_test>=Tstop*.3)-1.*(T_test>=Tstop*.7)).reshape(-1,1) # input signal
x0_test = np.array([0.,0.])  # initial state

# Continuous time system:
# yddot = -0.5*ydot + 0.3*u - 0.2*y - 0.1*y**3 --> x1dot = x2
#                                                  x2dot = -0.5*x2 + 0.3*u - 0.2*x1 - 0.1*x1**3
def truesystem(x0, U, T):
    # system generating the training and test dataset    
    Tc, Uc = diffrax.rectilinear_interpolation(T,U.reshape(-1))
    input_fcn = diffrax.LinearInterpolation(ts=Tc, ys=Uc)    

    @jax.jit
    def state_fcn_true(t, x, args):
        # state update function xdot = state_fcn(t,x,u)
        return jnp.array([x[1], -0.5*x[1] + 0.3*input_fcn.evaluate(t) - 0.2*x[0] - 0.1*x[0]**3])

    @jax.jit
    def output_fcn_(t, x, args):
        return x[0]
    output_fcn_true = jax.jit(jax.vmap(output_fcn_, in_axes=(0,0,None)))

    @jax.jit
    def solve(x0):
        term = diffrax.ODETerm(state_fcn_true)
        solver = diffrax.Dopri5()
        t0 = T[0]
        t1 = T[-1]
        dt0 = (T[1]-T[0])/10.
        saveat = diffrax.SaveAt(ts=jnp.array(T))
        #saveat = SaveAt(ts=jnp.arange(t0,t1,dt0))
        sol = diffrax.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=None, saveat=saveat, max_steps=100000)
        return sol
    
    sol = solve(x0)
    X = sol.ys
    Y = output_fcn_true(sol.ts, X, None)
    return Y, X

Y_train, _ = truesystem(x0_train, U_train, T_train)

Ys_train, ymean, ygain = standard_scale(Y_train)
Us_train, umean, ugain = standard_scale(U_train)

Y_test, _ = truesystem(x0_test, U_test, T_test)
Ys_test = (Y_test-ymean)*ygain
Us_test = (U_test-umean)*ugain

Ys_train = Ys_train.reshape(-1,1)
Ys_test = Ys_test.reshape(-1,1)

# Perform system identification
def activation(x):
    return 1. / (1. + jnp.exp(-x))
    #return jnp.maximum(0., x)

@jax.jit
def state_fcn(x, u, t, params):
    # This computes dx/dt = f(x,u) for continuous-time systems
    A, B, C, W1, W2, W3, b1, b2 = params
    return A@x+B@u+W3@activation(W1@x+W2@u+b1)+b2

@jax.jit
def output_fcn(x, u, t, params):
    C = params[2]
    return C@x

nnx = 10  # number of hidden neurons in state-update function

def init_fcn(seed):
    np.random.seed(seed)
    A = -np.eye(nx) # start with a stable linear system
    B = 0.1*np.random.randn(nx, nu)
    C = 0.1*np.random.randn(ny, nx)
    W1 = 0.1*np.random.randn(nnx, nx)
    W2 = 0.5*np.random.randn(nnx, nu)
    W3 = 0.5*np.random.randn(nx, nnx)
    b1 = np.zeros(nnx)
    b2 = np.zeros(nx)
    return [A, B, C, W1, W2, W3, b1, b2]

# initialize model coefficients
model = CTModel(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn)
model.integration_options(ode_solver=diffrax.Heun())
model.init(params=init_fcn(seed=1))
# L2-regularization on initial state and model coefficients
model.loss(rho_x0=1.e-4, rho_th=1.e-4, train_x0=True) # default: integral of squared output prediction errors
# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=100, lbfgs_epochs=2000, adam_eta=0.1)

# Fit model to training data
# model.fit(Ys_train, Us_train, T_train)

# Train from different initializations:
models = model.parallel_fit(Ys_train, Us_train, T_train, init_fcn=init_fcn)
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
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(T_train, Us_train, color=[.75,.75,.75], linestyle='--', linewidth=1., label='Input')
ax[0].plot(T_train, Ys_train, label='True output')
ax[0].plot(T_train, Yshat_train, label='Predicted output')
ax[0].set_title('Training data')
ax[0].legend()
ax[0].grid()
ax[0].set_xlabel('Time')
ax[1].plot(T_test, Us_test, color=[.75,.75,.75], linestyle='--', linewidth=1., label='Input')
ax[1].plot(T_test, Ys_test, label='True output')
ax[1].plot(T_test, Yshat_test, label='Predicted output')
ax[1].set_title('Test data')
ax[1].legend()
ax[1].grid()
ax[1].set_xlabel('Time')
