# -*- coding: utf-8 -*-
"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression/classification using JAX.

(C) 2024-2026 A. Bemporad
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax_sysid.utils import lbfgs_options, compute_scores
from joblib import Parallel, delayed, cpu_count

EPSIL_LASSO = 1.e-16  # tolerance used in groupLassoReg functions to prevent 0/0 = nan in the Jacobian vector when the argument is the zero vector
DEFAULT_SMALL_TAU_TH = 1.e-8  # add some small L1-regularization.
# This is parameter epsilon in Lemma 2 of the paper
# A. Bemporad, "Linear and nonlinear system identification under L1- and group-Lasso regularization via L-BFGS-B", submitted for publication and available on arXiv, 2024.

@jax.jit
def l2reg(z):
    """
    Compute the L2 regularization cost for a given set of parameters.

    Parameters
    ----------
    z : list
        List of parameters.

    Returns
    -------
    float
        The L2 regularization cost.
    """
    cost = 0.
    for w in z:
        cost += jnp.sum(w ** 2)
    return cost


@jax.jit
def l1reg(z):
    """
    Compute the L1 regularization cost for a given set of parameters.

    Parameters
    ----------
    z : list
        List of parameters.

    Returns
    -------
    float
        The L1 regularization cost.
    """
    cost = 0.
    for w in z:
        cost += jnp.sum(jnp.abs(w))
    return cost


@jax.jit
def linreg(z):
    """
    Compute the linear cost given by the sum of all parameters.

    Parameters
    ----------
    z : list
        List of parameters.

    Returns
    -------
    float
        The linear cost.
    """
    cost = 0.
    for w in z:
        cost += jnp.sum(w)
    return cost


def optimization_base(object, adam_eta=0.001, adam_epochs=0, lbfgs_epochs=1000, iprint=50, memory=10, lbfgs_tol=1.e-16):
    """Define the optimization parameters for the system identification problem.

    Parameters
    ----------
    object : object
        Model object to be trained by optimization.
    adam_eta : float
        Adam's learning rate (not used by LBFGS).
    adam_epochs : int
        Number of initial Adam iterations (adam_epochs=0 means pure L-BFGS-B).
    lbfgs_epochs : int
        Max number of function evaluations in the following L-BFGS iterations (lbfgs_epochs=0 means pure Adam).
    iprint : int
        How often printing L-BFGS updates (-1 = no printing, not even Adam iterations).
    memory : int
        L-BGFS memory (not used if method = 'Adam').
    lbfgs_tol : float
        Tolerance for L-BFGS-B iterations.
    Returns
    -------
    object
        The model object modified with the set of optimization parameters.
    """
    object.adam_eta = adam_eta
    object.adam_epochs = adam_epochs
    object.lbfgs_epochs = lbfgs_epochs
    object.iprint = iprint
    object.memory = memory
    object.lbfgs_tol = lbfgs_tol
    return object


def adam_solver(JdJ, z, solver_iters, adam_eta, iprint, params_min=None, params_max=None):
    """
    Solves a nonlinear optimization problem using Adam, with best-iterate tracking.
    JIT-compiled via jax.lax.scan. Uses the same hyperparameters as adam_solver
    (beta1=0.9, beta2=0.99) and returns the best point seen across all iterations.
    """
    beta1 = 0.9
    beta2 = 0.99
    epsil = 1e-8
    nz = len(z)

    ismin = (params_min is not None)
    ismax = (params_max is not None)
    isbounded = ismin or ismax

    z = [jnp.asarray(zi) for zi in z]
    if isbounded:
        for j in range(nz):
            if ismin:
                z[j] = jnp.maximum(z[j], params_min[j])
            if ismax:
                z[j] = jnp.minimum(z[j], params_max[j])

    m = [jnp.zeros_like(zi) for zi in z]
    v = [jnp.zeros_like(zi) for zi in z]

    if iprint > 0:
        _bar_total = solver_iters
        _bar_width = 20

        def _adam_bar(t, val, fbest):
            t = int(t)
            frac = (t + 1) / _bar_total
            filled = int(_bar_width * frac)
            bar = '█' * filled + '░' * (_bar_width - filled)
            print(f"\rAdam: {100*frac:3.0f}%|{bar}| {t+1}/{_bar_total} [loss={float(val):.4e}, best={float(fbest):.4e}]",
                  end='\033[K', flush=True)

        @jax.jit
        def scan_step(carry, t):
            z_, m_, v_, beta1t, beta2t, fbest_, zbest_ = carry
            val, g = JdJ(z_)
            m_ = [beta1 * mi + (1 - beta1) * gi for mi, gi in zip(m_, g)]
            v_ = [beta2 * vi + (1 - beta2) * gi**2 for vi, gi in zip(v_, g)]
            z_new = [zi - adam_eta / (1 - beta1t) * mi / (jnp.sqrt(vi / (1 - beta2t)) + epsil)
                     for zi, mi, vi in zip(z_, m_, v_)]
            if isbounded:
                for j in range(nz):
                    if ismin:
                        z_new[j] = jnp.maximum(z_new[j], params_min[j])
                    if ismax:
                        z_new[j] = jnp.minimum(z_new[j], params_max[j])
            zbest_ = jax.lax.cond(val < fbest_, lambda: z_, lambda: zbest_)
            fbest_ = jnp.minimum(val, fbest_)
            beta1t = beta1t * beta1
            beta2t = beta2t * beta2
            jax.lax.cond(
                (t % iprint == 0) | (t == _bar_total - 1),
                lambda _: jax.debug.callback(_adam_bar, t, val, fbest_, ordered=True),
                lambda _: None,
                operand=None,
            )
            return (z_new, m_, v_, beta1t, beta2t, fbest_, zbest_), val

        (_, _, _, _, _, Jopt, zbest), _ = jax.lax.scan(
            scan_step,
            (z, m, v, jnp.array(beta1), jnp.array(beta2), jnp.array(jnp.inf), z),
            jnp.arange(solver_iters))
    else:
        @jax.jit
        def scan_step(carry, _):
            z_, m_, v_, beta1t, beta2t, fbest_, zbest_ = carry
            val, g = JdJ(z_)
            m_ = [beta1 * mi + (1 - beta1) * gi for mi, gi in zip(m_, g)]
            v_ = [beta2 * vi + (1 - beta2) * gi**2 for vi, gi in zip(v_, g)]
            z_new = [zi - adam_eta / (1 - beta1t) * mi / (jnp.sqrt(vi / (1 - beta2t)) + epsil)
                     for zi, mi, vi in zip(z_, m_, v_)]
            if isbounded:
                for j in range(nz):
                    if ismin:
                        z_new[j] = jnp.maximum(z_new[j], params_min[j])
                    if ismax:
                        z_new[j] = jnp.minimum(z_new[j], params_max[j])
            zbest_ = jax.lax.cond(val < fbest_, lambda: z_, lambda: zbest_)
            fbest_ = jnp.minimum(val, fbest_)
            beta1t = beta1t * beta1
            beta2t = beta2t * beta2
            return (z_new, m_, v_, beta1t, beta2t, fbest_, zbest_), val

        (_, _, _, _, _, Jopt, zbest), _ = jax.lax.scan(
            scan_step,
            (z, m, v, jnp.array(beta1), jnp.array(beta2), jnp.array(jnp.inf), z),
            None, length=solver_iters)

    return list(zbest), float(Jopt)

def lbfgs_callback(iters, iprint, loss):
    cb = None
    cb_final = None
    if iprint > 0:
        _bar_total = iters
        _bar_width = 20
        _it = [0]

        def cb(x):
            _it[0] += 1
            it = _it[0]
            if it % iprint == 0 or it >= _bar_total:
                val   = float(loss(x))
                frac  = it / _bar_total
                filled = int(_bar_width * frac)
                bar   = '█' * filled + '░' * (_bar_width - filled)
                print(f"\rL-BFGS-B: {100*frac:3.0f}%|{bar}| {it}/{_bar_total} [loss={val:.4e}]",
                        end='\033[K', flush=True)

        def cb_final(val):
            it = _it[0]
            if it < _bar_total:
                bar = '█' * _bar_width
                print(f"\rL-BFGS-B: 100%|{bar}| {it}/{_bar_total} [loss={val:.4e}]",
                      end='\033[K', flush=True)

    return cb, cb_final

@jax.jit
def xsat(x, sat):
    """
    Apply saturation to the state value.

    Parameters
    ----------
    x : jax.numpy.ndarray
        The state value.
    sat : float
        The saturation limit.

    Returns
    -------
    jax.numpy.ndarray
        The saturated value of x.
    """
    return jnp.minimum(jnp.maximum(x, -sat), sat)  # hard saturation


def get_bounds(z, EPSIL_LASSO, params_min, params_max):
    """
    Utility function to create bounds for L-BFGS-B when splitting positive and negative parts

    Parameters
    ----------
    z : list
        List of parameters.
    EPSIL_LASSO : float
        Small positive value used to constrain the positive and negative parts to be strictly positive.
    params_min : list of arrays
        List of the same structure as self.params of lower bounds for the parameters.
    params_max : list of arrays
        List of the same structure as self.params of upper bounds for the parameters.
    """

    isbounded = (params_min is not None) or (params_max is not None)
    nz = len(z)

    lb = list()
    ub = list()
    if not isbounded:
        for h in range(2):
            for i in range(nz):
                lb.append(np.zeros_like(z[i])+EPSIL_LASSO)
                ub.append(np.inf*np.ones_like(z[i]))
    else:
        # We have bounds on the parameters:
        #     lb <= x  <= ub with y,z>=EPSIL_LASSO and x = y-z
        #
        # Then, we impose:
        #       max(0,lb) +EPSIL_LASSO <= y <= max(0,ub) +EPSIL_LASSO
        #       max(0,-ub)+EPSIL_LASSO <= z <= max(0,-lb)+EPSIL_LASSO
        #
        # Note: we could eliminate some variables when the lower and upper bounds
        # are both equal to EPSIL_LASSO
        for i in range(nz):
            # positive part y
            zi = jnp.zeros_like(z[i])
            lb.append(jnp.maximum(params_min[i], zi)+EPSIL_LASSO)
            ub.append(jnp.maximum(params_max[i], zi)+EPSIL_LASSO)
        for i in range(nz):
            # negative part z
            zi = jnp.zeros_like(z[i])
            lb.append(jnp.maximum(-params_max[i], zi)+EPSIL_LASSO)
            ub.append(jnp.maximum(-params_min[i], zi)+EPSIL_LASSO)
    return (lb, ub)


def find_best_model(models, Y, U, fit='R2', n_jobs=None, verbose=True):
    """
    Given a list of models, find the model that achieves the highest fit on a given dataset.

    Parameters
    ----------
    models : list
        List of models to evaluate.
    Y : np.ndarray
        Output data.
    U : np.ndarray
        Input data.
    fit : str or function, optional
        Metric to use for evaluating the fit. Default is 'R2', for other supported metrics
        (such as 'BFR', or 'RMSE', or 'Accuracy') see function utils/compute_scores().
        Alternatively, fit is a function fit(Y,Yhat) that takes two arguments: Y (=output data)
        and Yhat (=outputs predicted by each model) and returns a scalar fit value. The larger the
        value, the better the fit.
    n_jobs : int, optional
        Number of parallel jobs to run (default is None, which means using all available cores).
    verbose : bool
        If True, print the scores for each model.

    Returns
    -------
    model
        The model that achieves the highest fit (or highest average fit, in case of multiple targets).
    score
        The score of the best model.
    """
    # Import here to avoid circular imports; these classes are defined in their own modules
    from jax_sysid._ct_model import CTModel
    from jax_sysid._model import Model
    from jax_sysid._static_model import StaticModel

    if not isinstance(models, list):
        raise Exception(
            "\033[1mPlease provide a list of models to compare.\033[0m")

    # Recognize type of model
    if isinstance(models[0], CTModel):
        raise Exception(
            "\033[1mUse model.find_best_model(models,Y, U, fit, n_jobs, T, Tu, interpolation_type, ode_solver, dt0, max_steps, stepsize_controller\033[0m")

    if isinstance(fit, str):
        def get_score(Y,Yhat):
            score, _, _ = compute_scores(Y, Yhat, fit=fit)
            if fit.lower() == 'rmse':
                score = -score  # minimize RMSE
            return score
    else:
        get_score = fit

    def single_score(k):
        if isinstance(models[0], Model):
            x0 = models[k].learn_x0(U, Y)
            # use base class method to override possible redefitions of predict in subclasses
            Yhat, _ = Model.predict(models[k], x0, U)
        elif isinstance(models[0], StaticModel):
            Yhat = models[k].predict(U)
        else:
            raise Exception(
                "\033[1mUnknown model type\033[0m")
        score = get_score(Y, Yhat)
        return score

    if len(models) == 1:
        return models[0], single_score(0)  # if only one model, return it and its score directly

    if n_jobs is None:
        n_jobs = cpu_count()  # Use all available cores by default

    if verbose:
        _bar_total = len(models)
        _bar_width = 20
        scores = []
        for k in range(_bar_total):
            scores.append(single_score(k))
            frac = (k + 1) / _bar_total
            filled = int(_bar_width * frac)
            bar = '█' * filled + '░' * (_bar_width - filled)
            print(f"\rEvaluating models: {100*frac:3.0f}%|{bar}| {k+1}/{_bar_total}", end='\033[K', flush=True)
        print()
    else:
        scores = Parallel(n_jobs=n_jobs)(delayed(single_score)(k)
                                         for k in range(len(models)))

    best_id = np.argmax(np.sum(np.array(scores).reshape(len(models),-1),axis=1)) # get best score (best average score in case of multiple outputs)

    if isinstance(fit, str):
        fit_name = fit
        if fit.lower() == 'rmse':
            scores = [-s for s in scores]
    else:
        fit_name = fit.__name__

    if verbose:
        print(f"Best model: {best_id}, {fit_name} = {np.array(scores[best_id])}")

    return models[best_id], np.array(scores[best_id])
