"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

Linear system identification example in continuous time

            dx(t)/dt = A x(t) + B u(t)
                y(t) = C x(t) + D u(t)
                x(0) = x0

(C) 2024 A. Bemporad, December 22, 2024
"""

from jax_sysid.utils import standard_scale, compute_scores
from jax_sysid.models import CTModel
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import diffrax
import control as ctrl

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

seed = 3  # for reproducibility of results
np.random.seed(seed)

s = ctrl.TransferFunction.s
sys = (s + 1)/(s**3 + 3*s**2 +2*s + 1)
A, B, C, D = ctrl.ssdata(sys)
nx,nu = B.shape # number of states and inputs
ny = C.shape[0] # number of outputs

# Data generation
def generate_data(Tstop, Ts_u, Ts_y, umin, umax, y_noise, seed):
    Tu = np.arange(0.,Tstop+Ts_u, Ts_u) # add one more sample to define the interpolation over the entire range of time [0,Tstop]
    T = np.arange(0.,Tstop, Ts_y) # time at which the output is evaluated

    # random input excitation
    np.random.seed(seed)
    u0=umin+(umax-umin)*np.random.rand()
    U = np.empty(Tu.shape).reshape(-1,1)
    for i in range(Tu.size):
        if np.random.randn()>0.95:
            u0=umin+(umax-umin)*np.random.rand()
        U[i] = u0

    # Evaluate the input signal at the time instants T
    Tc, Uc = diffrax.rectilinear_interpolation(Tu, U.reshape(-1))
    input_eval = diffrax.LinearInterpolation(ts=Tc, ys=Uc)
    U_sim = input_eval.evaluate(T).reshape(-1)
    
    # Simulate the linear system
    sim = ctrl.forced_response(sys, T, U_sim)
    Y=sim.y.reshape(-1,1)
    T=sim.t.reshape(-1)
    
    # Add noise
    Y += y_noise*np.random.randn(Y.size).reshape(-1,1) # add measurement noise
    return Y, T, U, Tu

Y_train, T_train, U_train, Tu_train = generate_data(1000., 5., 1., -2., 2., 0.1, 1)
Y_test, T_test, U_test, Tu_test = generate_data(1000., 5., 1., -2., 2., 0.1, 2)

# Standardize data
Ys_train, ymean, ygain = standard_scale(Y_train)
Us_train, umean, ugain = standard_scale(U_train)
Ys_test = (Y_test-ymean)*ygain
Us_test = (U_test-umean)*ugain

Us_train = Us_train.reshape(-1,1)
Ys_train = Ys_train.reshape(-1,1)
Us_test = Us_test.reshape(-1,1)
Ys_test = Ys_test.reshape(-1,1)

@jax.jit
def state_fcn(x, u, t, params):
    # This computes dx/dt = f(x,u) for continuous-time systems
    A, B, C = params
    return A@x+B@u

@jax.jit
def output_fcn(x, u, t, params):
    A, B, C = params
    return C@x

def init_fcn(seed):
    np.random.seed(seed)
    A = -np.eye(nx) # start with a stable linear system
    B = 0.1*np.random.randn(nx, nu)
    C = 0.1*np.random.randn(ny, nx)
    return [A, B, C]

# initialize model coefficients
model = CTModel(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn)
model.integration_options(ode_solver=diffrax.Heun())
model.init(params=init_fcn(seed=1))
# L2-regularization on initial state and model coefficients
model.loss(rho_x0=1.e-4, rho_th=1.e-4, train_x0=True)
# number of epochs for Adam and L-BFGS-B optimization
model.optimization(adam_epochs=100, lbfgs_epochs=2000, adam_eta=0.1)

# Train from different initializations:
model.fit(Ys_train, Us_train, T_train, Tu_train)

x0_train=model.x0 # trained initial state
Yshat_train, _ = model.predict(x0_train, Us_train, T_train, Tu_train)

x0_test=model.learn_x0(Us_test, Ys_test, T_test, Tu_test)
Yshat_test, _ = model.predict(x0_test, Us_test, T_test, Tu_test)

# Compute scores
R2_train, R2_test, msg = compute_scores(Ys_train, Yshat_train, Ys_test, Yshat_test)
print(msg)

Ah, Bh, Ch = model.params
Dh = np.zeros((ny, nu))
sys_hat = ctrl.tf(ctrl.ss(Ah, Bh, Ch, Dh))

print(f"True system's DC gain       = {ctrl.dcgain(sys)}")
print(f"Identified system's DC gain = {ctrl.dcgain(sys_hat)}")

# Plot results
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(Tu_train, Us_train, color=[.75,.75,.75], linestyle='--', linewidth=1., label='Input')
ax[0].plot(T_train, Ys_train, label='True output')
ax[0].plot(T_train, Yshat_train, label='Predicted output')
ax[0].set_title('Training data')
ax[0].legend()
ax[0].grid()
ax[0].set_xlabel('Time')
ax[1].plot(Tu_test, Us_test, color=[.75,.75,.75], linestyle='--', linewidth=1., label='Input')
ax[1].plot(T_test, Ys_test, label='True output')
ax[1].plot(T_test, Yshat_test, label='Predicted output')
ax[1].set_title('Test data')
ax[1].legend()
ax[1].grid()
ax[1].set_xlabel('Time')
