# -*- coding: utf-8 -*-
"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression/classification using JAX.

(C) 2024-2026 A. Bemporad
"""

import jax
import jax.numpy as jnp
from ._model import Model


class RNN(Model):
    def __init__(self, nx, ny, nu, FX, FY=None, y_in_x=False, x_scaling=0.1, y_scaling=1.0, x0=None, seed=0, Ts=None):
        """Create a recurrent neural network model structure

            x(k+1) = fx(x(k),u(k))
            y(k) = fy(x(k),u(k))

        Parameters
        ----------
        nx : int
            Number of states.
        ny : int
            Number of outputs.
        nu : int
            Number of inputs.
        FX : subclass of flax.linen.nn.Module
            Feedforward neural-network function of the state update
        FY : subclass of flax.linen.nn.Module
            Feedforward neural-network function of the output
        y_in_x : bool
            If True, the output is included in the first ny components of the state vector and FY is not used
        x_scaling : float
            Scaling factor for the initial weight matrices of the state update function
        y_scaling : float
            Scaling factor for the weight matrices of the output function
        x0 : ndarray or list of ndarrays or None
            Initial state vector or list of initial state vectors for multiple experiments. If None, x0=0.
        seed : int
            Random seed for initializing the parameters of the neural networks (default: 0)
        Ts : float
            Sampling time
        """

        super().__init__(nx, ny, nu, y_in_x=y_in_x, Ts=Ts)

        key1, key2 = jax.random.split(jax.random.PRNGKey(seed), 2)

        self.fx = FX()
        # initialize parameters by passing a template vector
        thx = self.fx.init(key1, jnp.ones(nx+nu))['params']
        thx_flat, thx_tree = jax.tree_util.tree_flatten(thx)
        params = thx_flat
        self.nthx = len(thx_flat)
        self.thx_tree = thx_tree

        @jax.jit
        def state_fcn(x, u, params):
            thx = jax.tree_util.tree_unflatten(
                self.thx_tree, params[0:self.nthx])
            x = self.fx.apply({'params': thx}, jnp.hstack((x, u)))  # time update
            return x

        if not y_in_x:
            self.fy = FY()
            thy = self.fy.init(key2, jnp.ones(nx+nu))['params']
            thy_flat, thy_tree = jax.tree_util.tree_flatten(thy)
            params.extend(thy_flat)
            self.thy_tree = thy_tree

            @jax.jit
            def output_fcn(x, u, params):
                thy = jax.tree_util.tree_unflatten(
                    self.thy_tree, params[self.nthx:])
                y = self.fy.apply({'params': thy}, jnp.hstack(
                    (x, u)))  # predicted output
                return y
        else:
            @jax.jit
            def output_fcn(x, u, params):
                return x[0:self.ny]

        for i in range(len(params)):
            if i < self.nthx:
                params[i] *= x_scaling
            else:
                params[i] *= y_scaling

        self.state_fcn = state_fcn
        self.output_fcn = output_fcn
        self.params = params
        if x0 is not None:
            if isinstance(x0, list):
                Nexp = len(x0)
                self.x0 = [jnp.array(x0[i]) for i in range(Nexp)]
            else:
                self.x0 = jnp.array(x0)
        else:
            self.x0 = jnp.zeros(self.nx)
        self.isLinear = False
        self.isInitialized = True
        return

    def init_fcn(self,seed):
        """Initialize the model parameters given a seed.

        Parameters
        ----------
        seed : int
            Random seed for initializing the parameters.

        Returns
        -------
        params : list
            The initial guess for the model parameters.
        """
        key1, key2 = jax.random.split(jax.random.PRNGKey(seed), 2)
        thx = self.fx.init(key1, jnp.ones(self.nx+self.nu))['params']
        thx_flat, _ = jax.tree_util.tree_flatten(thx)
        params = thx_flat

        if not self.y_in_x:
            thy = self.fy.init(key2, jnp.ones(self.nx+self.nu))['params']
            thy_flat, _ = jax.tree_util.tree_flatten(thy)
            params.extend(thy_flat)

        return params
