# -*- coding: utf-8 -*-
"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression/classification using JAX.

(C) 2024-2026 A. Bemporad

This module re-exports the full public API from the individual sub-modules so
that existing code from earlier versions using ``from jax_sysid.models import ...`` 
continues to work without any changes.
"""

# Constants
from ._solvers import EPSIL_LASSO, DEFAULT_SMALL_TAU_TH

# Top-level helper functions
from ._solvers import (
    l2reg,
    l1reg,
    linreg,
    optimization_base,
    adam_solver,
    lbfgs_callback,
    xsat,
    get_bounds,
    find_best_model,
)

# Classes
from ._model import Model
from ._linear_model import LinearModel
from ._qlpv_model import qLPVModel
from ._rnn import RNN
from ._ct_model import CTModel
from ._static_model import StaticModel
from ._fnn import FNN

__all__ = [
    # Constants
    "EPSIL_LASSO",
    "DEFAULT_SMALL_TAU_TH",
    # Functions
    "l2reg",
    "l1reg",
    "linreg",
    "optimization_base",
    "adam_solver",
    "lbfgs_callback",
    "xsat",
    "get_bounds",
    "find_best_model",
    # Classes
    "Model",
    "LinearModel",
    "qLPVModel",
    "RNN",
    "CTModel",
    "StaticModel",
    "FNN",
]
