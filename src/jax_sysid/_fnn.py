# -*- coding: utf-8 -*-
"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression/classification using JAX.

(C) 2024-2026 A. Bemporad
"""

import jax
import jax.numpy as jnp
from ._static_model import StaticModel


class FNN(StaticModel):
    def __init__(self, ny, nu, FY, seed=0):
        """Create a feedforward neural network

            y(k) = f(x(k))

        Parameters
        ----------
        ny : int
            Number of outputs
        nu : int
            Number of inputs
        FY : subclass of flax.linen.nn.Module
            Feedforward neural-network function of the output
        seed : int
            Random seed for initializing the parameters of the neural network (default: 0)
        """

        key = jax.random.PRNGKey(seed)

        # initialize parameters by passing a template vector
        self.fy = FY()
        thy = self.fy.init(key, jnp.ones(nu))['params']
        thy_flat, thy_tree = jax.tree_util.tree_flatten(thy)
        params = thy_flat
        self.thy_tree = thy_tree

        @jax.jit
        def output_fcn(u, params):
            thy = jax.tree_util.tree_unflatten(self.thy_tree, params)
            y = self.fy.apply({'params': thy}, u)  # predicted output
            return y

        super().__init__(ny, nu, output_fcn=output_fcn)

        self.params = params
        self.isInitialized = True
        return

    def init_fcn(self, seed):
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
        key = jax.random.PRNGKey(seed)
        thy = self.fy.init(key, jnp.ones(self.nu))['params']
        thy_flat, _ = jax.tree_util.tree_flatten(thy)
        return thy_flat
