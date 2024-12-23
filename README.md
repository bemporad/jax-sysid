<img src="http://cse.lab.imtlucca.it/~bemporad/jax-sysid/images/jax-sysid-logo.png" alt="jax-sysid" width=40%/>

A Python package based on <a href="https://jax.readthedocs.io"> JAX </a> for linear and nonlinear system identification of state-space models, recurrent neural network (RNN) training, and nonlinear regression/classification.
 
# Contents

- [Contents](#contents)
  - [Package description](#package-description)
  - [Installation](#installation)
  - [Basic usage](#basic-usage)
    - [Linear state-space models](#linear-state-space-models)
      - [Training linear models](#training-linear-models)
      - [L1- and group-Lasso regularization](#l1--and-group-lasso-regularization)
      - [Multiple experiments](#multiple-experiments)
      - [Stability](#stability)
      - [Static gain](#static-gain)
    - [Nonlinear system identification and RNNs](#nonlinear-system-identification-and-rnns)
      - [Training nonlinear models](#training-nonlinear-models)
      - [Parallel training](#parallel-training)
      - [flax.linen models](#flaxlinen-models)
      - [Custom output loss](#custom-output-loss)
      - [Custom regularization](#custom-regularization)
      - [Static gain](#static-gain-1)
      - [Upper and lower bounds](#upper-and-lower-bounds)
    - [Quasi-Linear Parameter-Varying (qLPV) models](#quasi-linear-parameter-varying-qlpv-models)
      - [Training qLPV models](#training-qlpv-models)
    - [Continuous-time models](#continuous-time-models)
      - [Training continuous-time models](#training-continuous-time-models)
    - [Static models](#static-models)
      - [Nonlinear regression](#nonlinear-regression)
      - [Classification](#classification)
  - [Contributors](#contributors)
  - [Acknowledgments](#acknowledgments)
  - [Citing jax-sysid](#citing-jax-sysid)
  - [License](#license)


<a name="description"></a>
## Package description 

**jax-sysid** is a Python package based on <a href="https://jax.readthedocs.io"> JAX </a> for linear and nonlinear system identification of state-space models, recurrent neural network (RNN) training, and nonlinear regression/classification. The algorithm can handle L1-regularization and group-Lasso regularization and relies on L-BFGS optimization for accurate modeling, fast convergence, and good sparsification of model coefficients.

The package implements the approach described in the following paper:

<a name="cite-Bem24"></a>
> [1] A. Bemporad, "[Linear and nonlinear system identification under $\ell_1$- and group-Lasso regularization via L-BFGS-B](
http://arxiv.org/abs/2403.03827)," submitted for publication. Available on arXiv at <a href="http://arxiv.org/abs/2403.03827">
http://arxiv.org/abs/2403.03827</a>, 2024. [[bib entry](#ref1)]


<a name="install"></a>
## Installation

~~~python
pip install jax-sysid
~~~

<a name="basic-usage"></a>
## Basic usage

<a name="linear"></a>
### Linear state-space models
#### Training linear models
Given input/output training data $(u_0,y_0)$, $\ldots$, $(u_{N-1},y_{N-1})$, $u_k\in R^{n_u}$, $y_k\in R^{n_y}$, we want to identify a state-space model in the following form

$$        x_{k+1}=Ax_k+Bu_k$$

$$        \hat y_k=Cx_k+Du_k $$

where $k$ denotes the sample instant, $x_k\in R^{n_x}$ is the vector of hidden states, and
$A,B,C,D$ are matrices of appropriate dimensions to be learned.

The training problem to solve is

$$\min_{z}r(z)+\frac{1}{N}\sum_{k=0}^{N-1} \|y_{k}-Cx_k-Du_k\|_2^2$$

$$\textrm{s.t.}\ x_{k+1}=Ax_k+Bu_k, \ k=0,\ldots,N-2$$

where $z=(\theta,x_0)$ and $\theta$ collecting the entries of $A,B,C,D$.

The regularization term $r(z)$ includes the following components:

$$\rho_{\theta} \|\theta\|_2^2 $$

$$\rho_{x_0} \|x_0\|_2^2$$

$$\tau \left\|z\right\|_1$$

$$\tau_g\sum_{i=1}^{n_u} \|I_iz\|_2$$

with $\rho_\theta>0$, $\rho_{x_0}>0$, $\tau\geq 0$, $\tau_g\geq 0$. See examples below.

Let's start training a discrete-time linear model $(A,B,C,D)$ on a sequence of inputs $U=[u_0\ \ldots\ u_{N-1}]'$ and output $Y=[y_0\ \ldots\ y_{N-1}]'$, with regularization $\rho_\theta=10^{-2}$, $\rho_{x_0}=10^{-3}$, running the L-BFGS solver for at most 1000 function evaluations:

~~~python
from jax_sysid.models import LinearModel

model = LinearModel(nx, ny, nu)
model.loss(rho_x0=1.e-3, rho_th=1.e-2) 
model.optimization(lbfgs_epochs=1000) 
model.fit(Y,U)
Yhat, Xhat = model.predict(model.x0, U)
~~~

After identifying the model, to retrieve the resulting state-space realization you can use the following:

~~~python
A,B,C,D = model.ssdata()
~~~

Given a new test sequence of inputs and outputs, an initial state that is compatible with the identified model can be reconstructed by running an extended Kalman filter and Rauch–Tung–Striebel smoothing (cf. 
[[1]](#cite-Bem24)) and used to simulate the model:

~~~python
x0_test = model.learn_x0(U_test, Y_test)
Yhat_test, Xhat_test = model.predict(x0_test, U_test)
~~~

R2-scores on training and test data can be computed as follows:

~~~python
from jax_sysid.utils import compute_scores

R2_train, R2_test, msg = compute_scores(Y, Yhat, Y_test, Yhat_test, fit='R2')
print(msg)
~~~

It is good practice to scale the input and output signals. To identify a model on scaled signals, you can use the following:

~~~python
from jax_sysid.utils import standard_scale, unscale

Ys, ymean, ygain = standard_scale(Y)
Us, umean, ugain = standard_scale(U)
model.fit(Ys, Us)
Yshat, Xhat = model.predict(model.x0, Us)
Yhat = unscale(Yshat, ymean, ygain)
~~~

#### L1- and group-Lasso regularization
Let us now retrain the model using L1-regularization
and check the sparsity of the resulting model:

~~~python
model.loss(rho_x0=1.e-3, rho_th=1.e-2, tau_th=0.03) 
model.fit(Ys, Us)
print(model.sparsity_analysis())
~~~
                 
To reduce the number of states in the model, you can use group-Lasso regularization as follows:

~~~python
model.loss(rho_x0=1.e-3, rho_th=1.e-2, tau_g=0.1) 
model.group_lasso_x()
model.fit(Ys, Us)
~~~
Groups in this case are entries in $A,B,C,x_0$ related to the same state.

Group-Lasso can be also used to try to reduce the number of inputs that are relevant in the model. You can do this as follows:

~~~python
model.loss(rho_x0=1.e-3, rho_th=1.e-2, tau_g=0.15) 
model.group_lasso_u()
model.fit(Ys, Us)
~~~
Groups in this case are entries in $B,D$ related to the same input.

#### Multiple experiments
**jax-sysid** also supports multiple training experiments. In this case, the sequences of training inputs and outputs are passed as a list of arrays. For example, if three experiments are available for training, use the following command:

~~~python
model.fit([Ys1, Ys2, Ys3], [Us1, Us2, Us3])
~~~

In case the initial state $x_0$ is trainable, one initial state per experiment is optimized. To avoid training the initial state, add `train_x0=False` when calling `model.loss`.

#### Stability
To attempt forcing that the identified linear model is asymptotically stable, i.e., that matrix $A$ has all eigenvalues inside the unit disk, you can use the following command:

~~~python
model.force_stability()
~~~

before calling the `fit` function. This will introduce a custom regularization penalty that tries to enforce the constraint $\|A\|_2<1$.

#### Static gain
To introduce a penalty that attempts forcing the identified linear model to have a given DC-gain matrix `M`, you can use the following commands:

~~~python
dcgain_loss = model.dcgain_loss(DCgain = M)
model.loss(rho_x0=1.e-3, rho_th=1.e-2, custom_regularization = dcgain_loss)
~~~

before calling the `fit` function. Similarly, to fit instead the DC-gain of the model to steady-state input data `Uss` and corresponding output data `Yss`, you can use 

~~~python
dcgain_loss = model.dcgain_loss(Uss = Uss, Yss = Yss)
~~~
and use `dcgain_loss` as the custom regularization function.

<a name="nonlinear"></a>
### Nonlinear system identification and RNNs
#### Training nonlinear models
Given input/output training data $(u_0,y_0)$, $\ldots$, $(u_{N-1},y_{N-1})$, $u_k\in R^{n_u}$, $y_k\in R^{n_y}$, we want to identify a nonlinear parametric state-space model in the following form

$$        x_{k+1}=f(x_k,u_k,\theta)$$

$$        \hat y_k=g(x_k,u_k,\theta)$$

where $k$ denotes the sample instant, $x_k\in R^{n_x}$ is the vector of hidden states, and $\theta$ collects the trainable parameters of the model.

As for the linear case, the training problem to solve is

$$  \min_{z}r(z)+\frac{1}{N}\sum_{k=0}^{N-1} \|y_{k}-g(x_k,u_k,\theta)\|_2^2$$

$$\textrm{s.t.}\ x_{k+1}=f(x_k,u_k,\theta),\ k=0,\ldots,N-2$$

where $z=(\theta,x_0)$. The regularization term $r(z)$ is the same as in the linear case.

For example, let us consider the following residual RNN model without input/output feedthrough:

$$ x_{k+1}=Ax_k+Bu_k+f_x(x_k,u_k,\theta_x)$$ 

$$ \hat y_k=Cx_k+f_y(x_k,\theta_y)$$

where $f_x$, $f_y$ are feedforward shallow neural networks, and let $z$ collects the coefficients in $A,B,C,D,\theta_x,\theta_y$. We want to train $z$ by running 1000 Adam iterations followed by at most 1000 L-BFGS function evaluations:

~~~python
from jax_sysid.models import Model

Ys, ymean, ygain = standard_scale(Y)
Us, umean, ugain = standard_scale(U)

def sigmoid(x):
    return 1. / (1. + jnp.exp(-x))  
@jax.jit
def state_fcn(x,u,params):
    A,B,C,W1,W2,W3,b1,b2,W4,W5,b3,b4=params
    return A@x+B@u+W3@sigmoid(W1@x+W2@u+b1)+b2    
@jax.jit
def output_fcn(x,u,params):
    A,B,C,W1,W2,W3,b1,b2,W4,W5,b3,b4=params
    return C@x+W5@sigmoid(W4@x+b3)+b4

model = Model(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn)
nnx = 5 # number of hidden neurons in state-update function
nny = 5  # number of hidden neurons in output function

# Parameter initialization:
A  = 0.5*np.eye(nx)
B = 0.1*np.random.randn(nx,nu)
C = 0.1*np.random.randn(ny,nx)
W1 = 0.1*np.random.randn(nnx,nx)
W2 = 0.5*np.random.randn(nnx,nu)
W3 = 0.5*np.random.randn(nx,nnx)
b1 = np.zeros(nnx)
b2 = np.zeros(nx)
W4 = 0.5*np.random.randn(nny,nx)
W5 = 0.5*np.random.randn(ny,nny)
b3 = np.zeros(nny)
b4 = np.zeros(ny)
model.init(params=[A,B,C,W1,W2,W3,b1,b2,W4,W5,b3,b4]) 

model.loss(rho_x0=1.e-4, rho_th=1.e-4)
model.optimization(adam_epochs=1000, lbfgs_epochs=1000) 
model.fit(Ys, Us)
Yshat, Xshat = model.predict(model.x0, Us)
Yhat = unscale(Yshat, ymean, ygain)
~~~

#### Parallel training
As the training problem, in general, is a nonconvex optimization problem, the obtained model often depends on the initial value of the parameters. The **jax-sysid** library supports training models in parallel (including static models) using the `joblib` library. In the example above, we can train 10 different models using 10 jobs in `joblib` as follows:

~~~
def init_fcn(seed):
    np.random.seed(seed)
    A  = 0.5*np.eye(nx)
    B = 0.1*np.random.randn(nx,nu)
    C = 0.1*np.random.randn(ny,nx)
    W1 = 0.1*np.random.randn(nnx,nx)
    W2 = 0.5*np.random.randn(nnx,nu)
    W3 = 0.5*np.random.randn(nx,nnx)
    b1 = np.zeros(nnx)
    b2 = np.zeros(nx)
    W4 = 0.5*np.random.randn(nny,nx)
    W5 = 0.5*np.random.randn(ny,nny)
    b3 = np.zeros(nny)
    b4 = np.zeros(ny)
    return [A,B,C,W1,W2,W3,b1,b2,W4,W5,b3,b4]

models = model.parallel_fit(Ys, Us, init_fcn=init_fcn, seeds=range(10), n_jobs=10)
~~~

By default, `n_jobs` is equal to the number of all available CPUs. 

To select the best model on a dataset `Us_test`, `Ys_test` in accordance with a given fit criterion, you can use `find_best_model`:

~~~
from jax_sysid.models import find_best_model

best_model, best_R2 = find_best_model(models, Ys_test, Us_test, fit='R2')
~~~

#### flax.linen models
**jax-sysid** also supports recurrent neural networks defined via the **flax.linen** library (the `flax` package can be installed via `pip install flax`):

~~~python
from jax_sysid.models import RNN

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
model.loss(rho_x0=1.e-4, rho_th=1.e-4, tau_th=0.0001)
model.optimization(adam_epochs=0, lbfgs_epochs=2000) 
model.fit(Ys, Us)
~~~
where the extra parameter `x_scaling` is used to scale down (when $0\leq$ `x_scaling` $<1$) the default initialization of the network weights instantiated by **flax**.

#### Custom output loss
**jax-sysid** also supports custom loss functions penalizing the deviations of $\hat y$ from $y$. For example, to identify a system with a binary output, we can use the (modified) cross-entropy loss

$$
	{\mathcal L}(\hat Y,Y)=\frac{1}{N}\sum_{k=0}^{N-1}
	-y_k\log(\epsilon+\hat y_k)-(1-y_k)\log(\epsilon+1-\hat y_k)
$$

where $\hat Y=(\hat y_0,\ldots,\hat y_{N-1})$ and $Y=(y_0,\ldots, y_{N-1})$ are the sequences of predicted and measured outputs, respectively, and $\epsilon>0$ is a tolerance used to prevent numerical issues in case $\hat y_k\approx 0$ or $\hat y_k\approx 1$:

~~~python
epsil=1.e-4
@jax.jit
def cross_entropy_loss(Yhat,Y):
    loss=jnp.sum(-Y*jnp.log(epsil+Yhat)-(1.-Y)*jnp.log(epsil+1.-Yhat))/Y.shape[0]
    return loss
model.loss(rho_x0=0.01, rho_th=0.001, output_loss=cross_entropy_loss)
~~~

By default, **jax-sysid** minimizes the classical mean squared error

$$
	{\mathcal L}(\hat Y,Y)=\frac{1}{N}\sum_{k=0}^{N-1}
	\|y_k-\hat y_k\|_2^2
$$

#### Custom regularization
**jax-sysid** also supports custom regularization terms $r_c(z)$, where $z=(\theta,x_0)$. You can specify such a custom regularization function when defining the overall loss. For example, say for some reason you want to impose $\|\theta\|_2^2\leq 1$ as a soft constraint, you can penalize

$$\frac{1}{2} \rho_{\theta} \|\theta\|_2^2 + \rho_{x_0} \|x_0\|_2^2 + \rho_c\max\{\|\theta\|_2^2-1,0\}^2$$

with $\rho_c\gg\rho_\theta$, $\rho_c\gg\rho_{x_0}$, for instance $\rho_c=1000$, $\rho_\theta=0.001$, $\rho_{x0}=0.01$. In Python:

~~~python
@jax.jit
def custom_reg_fcn(th,x0):
    return 1000.*jnp.maximum(jnp.sum(th**2)-1.,0.)**2
model.loss(rho_x0=0.01, rho_th=0.001, custom_regularization= custom_reg_fcn)
~~~

#### Static gain
As for linear systems, a special case of custom regularization function to fit the DC-gain of the model to steady-state input data `Uss` and corresponding output data `Yss`  is obtained using the following commands:

~~~python
dcgain_loss = model.dcgain_loss(Uss = Uss, Yss = Yss)
model.loss(rho_x0=1.e-3, rho_th=1.e-2, custom_regularization = dcgain_loss)
~~~

before calling the `fit` function. Note that this penalty involves solving a system of nonlinear equations for every input/output steady-state pair to evaluate the loss function, so it can be slow if many steady-state data pairs are given.

#### Upper and lower bounds
To include lower and upper bounds on the parameters of the model and/or the initial state, use the following additional arguments when specifying the optimization problem:

~~~python
model.optimization(params_min=lb, params_max=ub, x0_min=xmin, x0_max=xmax, ...)
~~~

where `lb` and `ub` are lists of arrays with the same structure as `model.params`, while `xmin` and `xmax` are arrays of the same dimension `model.nx` of the state vector. By default, each value is set equal to `None`, i.e., the corresponding constraint is not enforced. See `example_linear_positive.py` for examples of how to use nonnegative constraints to fit a positive linear system.

<a name="quasiLPV"></a>
### Quasi-Linear Parameter-Varying (qLPV) models
#### Training qLPV models
As a special case of nonlinear dynamical models, **jax-sysid** supports the identification of quasi-LPV models of the form

$$x_{k+1} = A(p_k)x_k + B(p_k)u_k$$

$$y_k = C(p_k) x_k+D(p_k)u_k$$

where the scheduling vector $p_k$ is an arbitrary parametric nonlinear function (to be trained) of $x_k$ and $u_k$ 
with $n_p$ entries 

$$p_k = f(x_k,u_k,\theta_p)$$

and

$$A(p_k) = A_{\rm lin}+\sum_{i=1}^{n_p} A_i p_{ki},~~B(p_k) = B_{\rm lin}+\sum_{i=1}^{n_p} B_i p_{ki}$$

$$C(p_k) = C_{\rm lin}+\sum_{i=1}^{n_p} C_i p_{ki},~~D(p_k) = D_{\rm lin}+\sum_{i=1}^{n_p} D_i p_{ki}$$

For both LTI and qLPV models, **jax-sysid** must enable the feedthrough term $D(p_k)=0$ by specifying the flag `feedthrough=True` when defining the model (by default, no feedthrough is in place). Moreover, for all linear, nonlinear, and qLPV models, one can force $y_k=[I\ 0]x_k$ by specifying `y_in_x=True` in the object constructor.

Let's train a quasi-LPV model on a sequence of inputs $U=[u_0\ \ldots\ u_{N-1}]'$ and output $Y=[y_0\ \ldots\ y_{N-1}]'$, with scheduling parameter function $f(x,u,\theta_p)$ = `qlpv_fcn`, initial parameters $\theta_p$ = `qlpv_params_init`, regularization $\rho_\theta=10^{-2}$, $\rho_{x_0}=10^{-3}$, running the L-BFGS solver for at most 1000 function evaluations:

~~~python
from jax_sysid.models import qLPVModel

model = qLPVModel(nx, ny, nu, npar, qlpv_fcn, qlpv_params_init)

model.loss(rho_x0=1.e-3, rho_th=1.e-2) 
model.optimization(lbfgs_epochs=1000) 
model.fit(Ys, Us, LTI_training=True)
Yhat, Xhat = model.predict(model.x0, U)
~~~

where `Us`, `Ys` are the scaled input and output signals. The flag `LTI_training=True` forces the training algorithm to initialize $A_{\rm lin},B_{\rm lin},C_{\rm lin},D_{\rm lin}$
by first fitting an LTI model.

After identifying the model, to retrieve the resulting matrices $A_{\rm lin}$, $B_{\rm lin}$, $C_{\rm lin}$, $D_{\rm lin}$, $\{A_i\}$, $\{B_i\}$, $\{C_i\}$, $\{D_i\}$, $i=1,\ldots,n_p$, you can use the following:

~~~python
A, B, C, D, Ap, Bp, Cp, Dp = model.ssdata()
~~~

where `Ap`, `Bp`, `Cp`, `Dp` are tensors (3D matrices) containing the corresponding linear matrices, i.e., `Ap[i,:,:]`=$A_i$,
`Bp[i,:,:]`=$B_i$, `Cp[i,:,:]`=$C_i$, `Dp[i,:,:]`=$D_i$.

To attempt to reduce the number of scheduling variables in the model, you can use group-Lasso regularization as follows:

~~~python
model.loss(rho_x0=1.e-3, rho_th=1.e-2, tau_g=0.1) 
model.group_lasso_p()
model.fit(Ys, Us)
~~~
Each group $i=1,\ldots,n_p$ collects the entries of $(A_i, B_i, C_i, D_i)$ and `tau_g` is the weight associated with the corresponding group-Lasso penalty.

Parallel training from different initial guesses is also supported for qLPV models. To this end, you must define a function `qlpv_param_init_fcn(seed)` that initializes the parameter vector $\theta_p$ of the scheduling function for a given `seed`. For example,
you can train the model for 10 different random seeds on 10 jobs by running:

~~~python
models = model.parallel_fit(Y, U, qlpv_param_init_fcn=qlpv_param_init_fcn, seeds=range(10), n_jobs=10)
~~~

<a name="continuous-time"></a>
### Continuous-time models
#### Training continuous-time models
**jax-sysid** supports the identification of general parametric nonlinear continuous-time models

$$\frac{dx(t)}{dt} = f_x(x(t),u(t),t,\theta)$$
$$y_k = f_y(x(t),u(t),t,\theta)$$

by using the package `diffrax` (Kidger, 2021) for the integration of ordinary differential equations.

The default loss function is
$$\frac{1}{t_{end}-t_{init}}\int_{t_{init}}^{t_{end}}(\hat y(t)-y(t))^2 dt$$
where $t_{init}$ and $t_{end}$ are the initial and final time over which the training dataset is defined.

Linear time-invariant continuous-time models are a special case in which 
$$f_x(x(t),u(t),t,\theta)=Ax(t)+Bu(t)$$
$$f_y(x(t),u(t),t,\theta)=Cx(t)+Du(t)$$
with $\theta=(A,B,C,D)$.

You can train a continuous time model with `nx` states, `ny` outputs, and `nu` inputs as follows:

~~~
from jax_sysid import CTModel
model = CTModel(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn)
~~~

where `state_fcn(x, u, t, params)` defines the state update $\frac{dx(t)}{dt}$ and
`output_fcn(x, u, t, params)` the output $y(t)$.

Then, run

~~~
model.init(params) # initial values of the parameters
model.fit(Y, U, T)
~~~

By default, the integration method `diffrax.Heun()` is employed, with an integration step equal to (`T[1]-T[0]`)/10
and the input signal `U` is modeled in continuous-time by assuming that a zero-order hold (ZOH) keeps the
samples `U[k]` constant over each time interval in `T`.

To change integration options, such as to use a different integration solver like `diffrax.Euler()`, `diffrax.Dopri5()`, etc., you must call

~~~
model.integration_options(ode_solver=diffrax.Dopri5())
~~~

After training, predictions can be obtained by running

~~~
Y, X = model.predict(model.x0, U, T)
~~~

You can reconstruct an initial state on test data by running

~~~
x0_test = model.learn_x0(U_test, Y_test, T_test)
~~~

To specify that the input signal is sampled at different time instants, you must use an extra input argument to specify the array of time instants at which the input is sampled:

~~~
model.fit(Y, U, T, Tu)
x0_test = model.learn_x0(U_test, Y_test, T_test, Tu_test)
~~~

<a name="static"></a>
### Static models
#### Nonlinear regression
The same optimization algorithms used to train dynamical models can be used to train static models, i.e., to solve the nonlinear regression problem:

$$  \min_{z}r(z)+\frac{1}{N}\sum_{k=0}^{N-1} \|y_{k}-f(u_k,\theta)\|_2^2$$

where $z=\theta$ is the vector of model parameters to train and $r(z)$ admits the same
regularization terms as in the case of dynamical models.

For example, if the model is a shallow neural network you can use the following code:

~~~python
from jax_sysid.models import StaticModel
from jax_sysid.utils import standard_scale, unscale

@jax.jit
def output_fcn(u, params):
    W1,b1,W2,b2=params
    y = W1@u.T+b1
    y = W2@jnp.arctan(y)+b2
    return y.T
model = StaticModel(ny, nu, output_fcn)
nn=10 # number of neurons
model.init(params=[np.random.randn(nn,nu), np.random.randn(nn,1), np.random.randn(1,nn), np.random.randn(1,1)])
model.loss(rho_th=1.e-4, tau_th=tau_th) 
model.optimization(lbfgs_epochs=500) 
model.fit(Ys, Us)
~~~

**jax-sysid** also supports feedforward neural networks defined via the **flax.linen** library:

~~~python
from jax_sysid.models import FNN
from flax import linen as nn 

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
model.loss(rho_th=1.e-4, tau_th=tau_th)
model.optimization(lbfgs_epochs=500)
model.fit(Ys, Us)
~~~

To include lower and upper bounds on the parameters of the model, use the following additional arguments when specifying the optimization problem:

~~~python
model.optimization(lbfgs_epochs=500, params_min=lb, params_max=ub)
~~~

where `lb` and `ub` are lists of arrays with the same structure as `model.params`. See `example_static_convex.py` for examples of how to use nonnegative constraints to fit input-convex neural networks.

#### Classification
To solve classification problems, you need to define a custom loss function to change the default Mean-Squared-Error loss. For example, to train a classifier for a multi-category classification problem with $K$ classes, you can specify a neural network with a linear output layer generating output predictions $\hat y\in R^K$ and define the associated cross-entropy $\ell(\hat y,y) = -\sum_{k=1}^Ky_k\log\left(\frac{e^{\hat y_k}}{\sum_{j=1}^Ke^{\hat y_j}}\right)$ function as follows: 

~~~python
def cross_entropy(Yhat,Y):
    return -jax.numpy.sum(jax.nn.log_softmax(Yhat, axis=1)*Y)/Y.shape[0] 
model.loss(rho_th=1.e-4, output_loss=cross_entropy)
~~~

See `example_static_fashion_mist.py` for an example using **Keras** with JAX backend to define the neural network model.
                
<a name="contributors"></a>
## Contributors

This package was coded by Alberto Bemporad.


This software is distributed without any warranty. Please cite the paper below if you use this software.

<a name="acknowledgments"></a>
## Acknowledgments
We thank Roland Toth for suggesting the use of Kung's method for initializing linear state-space models and Kui Xie for feedback on the reconstruction of the initial state via EKF + RTS smoothing. 

<a name="bibliography"></a>
## Citing jax-sysid

<a name="ref1"></a>

```
@article{Bem24,
    author={A. Bemporad},
    title={Linear and nonlinear system identification under $\ell_1$- and group-{Lasso} regularization via {L-BFGS-B}},
    note = {submitted for publication. Also available on arXiv
    at \url{http://arxiv.org/abs/2403.03827}},
    year=2024
}
```

<a name="license"></a>
## License

Apache 2.0

(C) 2024 A. Bemporad
