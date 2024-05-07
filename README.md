<img src="http://cse.lab.imtlucca.it/~bemporad/jax-sysid/images/jax-sysid-logo.png" alt="jax-sysid" width=40%/>

A Python package based on <a href="https://jax.readthedocs.io"> JAX </a> for linear and nonlinear system identification of state-space models, recurrent neural network (RNN) training, and nonlinear regression/classification.
 
# Contents

* [Package description](#description)

* [Installation](#install)

* [Basic usage](#basic-usage)
    * [Linear state-space models](#linear)
    * [Nonlinear system identification and RNNs](#nonlinear)
    * [Static models and nonlinear regression/classification] (#static)

* [Contributors](#contributors)

* [Citing jax-sysid](#bibliography)

* [License](#license)


<a name="description"></a>
## Package description 

**jax-sysid** is a Python package based on <a href="https://jax.readthedocs.io"> JAX </a> for linear and nonlinear system identification of state-space models, recurrent neural network (RNN) training, and nonlinear regression/classification. The algorithm can handle L1-regularization and group-Lasso regularization and relies on L-BFGS optimization for accurate modeling, fast convergence, and good sparsification of model coefficients.

The package implements the approach described in the following paper:

<a name="cite-Bem24"><a>
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

Given input/output training data $(u_0,y_0)$, $\ldots$, $(u_{N-1},y_{N-1})$, $u_k\in R^{n_u}$, $y_k\in R^{n_y}$, we want to identify a state-space model in the following form

$$        x_{k+1}=Ax_k+Bu_k$$

$$        \hat y_k=Cx_k+Du_k $$

where $k$ denotes the sample instant, $x_k\in R^{n_x}$ is the vector of hidden states, and
$A,B,C,D$ are matrices of appropriate dimensions to be learned.

The training problem to solve is

$$\min_{z}r(z)+\frac{1}{N}\sum_{k=0}^{N-1} \|y_{k}-Cx_k-Du_k\|_2^2$$

$$\mbox{s.t.}\ x_{k+1}=Ax_k+Bu_k, \ k=0,\ldots,N-2$$

where $z=(\theta,x_0)$ and $\theta$ collecting the entries of $A,B,C,D$.

The regularization term $r(z)$ includes the following components:

$$\frac{1}{2} \rho_{\theta} \|\theta\|_2^2 $$

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

**jax-sysid** also supports multiple training experiments. In this case, the sequences of training inputs and outputs are passed as a list of arrays. For example, if three experiments are available for training, use the following command:

~~~python
model.fit([Ys1, Ys2, Ys3], [Us1, Us2, Us3])
~~~

In case the initial state $x_0$ is trainable, one initial state per experiment is optimized. To avoid training the initial state, add `train_x0=False` when calling `model.loss`.

<a name="nonlinear"></a>
### Nonlinear system identification and RNNs
Given input/output training data $(u_0,y_0)$, $\ldots$, $(u_{N-1},y_{N-1})$, $u_k\in R^{n_u}$, $y_k\in R^{n_y}$, we want to identify a nonlinear parametric state-space model in the following form

$$        x_{k+1}=f(x_k,u_k,\theta)$$

$$        \hat y_k=g(x_k,u_k,\theta)$$

where $k$ denotes the sample instant, $x_k\in R^{n_x}$ is the vector of hidden states, and $\theta$ collects the trainable parameters of the model.

As for the linear case, the training problem to solve is

$$  \min_{z}r(z)+\frac{1}{N}\sum_{k=0}^{N-1} \|y_{k}-g(x_k,u_k,\theta)\|_2^2$$

$$\mbox{s.t.}\ x_{k+1}=f(x_k,u_k,\theta),\ k=0,\ldots,N-2$$

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

**jax-sysid** also supports recurrent neural networks defined via the **flax.linen** library:


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

**jax-sysid** also supports custom regularization terms $r_c(z)$, where $z=(\theta,x_0)$. You can specify such a custom regularization function when defining the overall loss. For example, say for some reason you want to impose $\|\theta\|_2^2\leq 1$ as a soft constraint, you can penalize

$$\frac{1}{2} \rho_{\theta} \|\theta\|_2^2 + \rho_{x_0} \|x_0\|_2^2 + \rho_c\max\{\|\theta\|_2^2-1,0\}^2$$

with $\rho_c\gg\rho_\theta$, $\rho_c\gg\rho_{x_0}$, for instance $\rho_c=1000$, $\rho_\theta=0.001$, $\rho_{x0}=0.01$. In Python:

~~~python
@jax.jit
def custom_reg_fcn(th,x0):
    return 1000.*jnp.maximum(jnp.sum(th**2)-1.,0.)**2
model.loss(rho_x0=0.01, rho_th=0.001, custom_regularization= custom_reg_fcn)
~~~

To include lower and upper bounds on the parameters of the model and/or the initial state, use the following additional arguments when specifying the optimization problem:

~~~python
model.optimization(params_min=lb, params_max=ub, x0_min=xmin, x0_max=xmax, ...)
~~~

where `lb` and `ub` are lists of arrays with the same structure as `model.params`, while `xmin` and `xmax` are arrays of the same dimension `model.nx` of the state vector. By default, each value is set equal to `None`, i.e., the corresponding constraint is not enforced. See `example_linear_positive.py` for examples of how to use nonnegative constraints to fit a positive linear system.

<a name="static"></a>
### Static models and nonlinear regression / classification
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

To solve classification problems, you need to define a custom loss function to change the default Mean-Squared-Error loss. For example, to train a classifier for a multi-category classification problem with $K$ classes, you can specify a neural network with a linear output layer generating output predictions $\hat y\in R^K$ and define the associated cross-entropy $\ell(\hat y,y) = -\sum_{k=1}^Ky_k\log\left(\frac{e^{\hat y_k}}{\sum_{j=1}^Ke^{\hat y_j}}\right)$ function as follows: 

~~~python
def cross_entropy(Yhat,Y):
    return -jax.numpy.sum(jax.nn.log_softmax(Yhat, axis=1)*Y)/Y.shape[0] 
model.loss(rho_th=1.e-4, output_loss=cross_entropy)
~~~

See `example_static_fashion_mist.py` for an example using **Keras** with JAX backend to define the neural network model.
                
<a name="contributors"><a>
## Contributors

This package was coded by Alberto Bemporad.


This software is distributed without any warranty. Please cite the paper below if you use this software.

<a name="bibliography"><a>
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

<a name="license"><a>
## License

Apache 2.0

(C) 2024 A. Bemporad
