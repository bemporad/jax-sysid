# -*- coding: utf-8 -*-
"""
jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression/classification using JAX.

Utility functions.

(C) 2024 A. Bemporad, March 6, 2024
"""

import numpy as np
from scipy.linalg import svd, lstsq


def lbfgs_options(iprint, iters, lbfgs_tol, memory):
    """
    Create a dictionary of options for the L-BFGS-B optimization algorithm.

    Parameters
    ----------
    iprint : int
        Verbosity level for printing optimization progress.
    iters : int
        Maximum number of iterations.
    lbfgs_tol : float
        Tolerance for convergence.
    memory : int
        Number of previous iterations to store for computing the next search direction.

    Returns
    -------
    dict
        A dictionary of options for the L-BFGS-B algorithm.
    """
    options = {'iprint': iprint, 'maxls': 20, 'gtol': lbfgs_tol, 'eps': 1.e-8,
               'ftol': lbfgs_tol, 'maxfun': iters, 'maxcor': memory}
    return options


def standard_scale(X):
    """
    Standardize the input data by subtracting the mean and dividing by the standard deviation.
    Note that in the case of multi-output systems, scaling outputs changes the relative weight 
    of output errors.

    Parameters
    ----------
    X : ndarray
        Input data array of shape (n_samples, n_features).

    Returns
    -------
    standardized_data : ndarray
        The standardized data array of shape (n_samples, n_features).
    mean : ndarray
        The mean values of each feature.
    gain : ndarray
        The gain values used for scaling each feature (=inverse of std, if nonzero, otherwise 1).
    """
    X = vec_reshape(X)
    xmean = np.mean(X, 0)
    xstd = np.std(X, 0)
    nX = xmean.shape[0]
    xgain = np.ones(nX)
    for i in range(nX):
        if xstd[i] > 1.e-6:  # do not change gain in case std is approximately 0
            xgain[i] = 1./xstd[i]
    return (X - xmean) * xgain, xmean, xgain


def unscale(Xs, offset, gain):
    """
    Unscale scaled signal.

    Parameters
    ----------
    Xs : ndarray
        Scaled data array of shape (n_samples, n_features).
    offset : ndarray
        Offset to be added to each row of X.
    gain : ndarray
        Gain to be multiplied to each row of X.

    Returns
    -------
    X : ndarray
        Unscaled array X = Xs/gain + offset
    """
    return Xs/gain+offset


def vec_reshape(y):
    """Reshape an array to a 2D array if it is 1D.

    (C) 2024 A. Bemporad

    Parameters
    ----------
    y : ndarray
        Input array

    Returns:
    --------
    y : ndarray
        Possibly reshaped array
    """
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return y


def compute_scores(Y_train, Yhat_train, Y_test=None, Yhat_test=None, fit='R2'):
    """Compute R2-score, best fit rate, or accuracy score on (possibly multi-dimensional) 
       training and test output data

    (C) 2024 A. Bemporad

    Parameters
    ----------
    Y_train : ndarray or None
        Reference output data for training, with shape (n_samples_training, n_outputs)
    Yhat_train : ndarray or None
        Predicted output data for training, with shape (n_samples_training, n_outputs)
    Y_test : ndarray or None
        Reference output data for testing, with shape (n_samples_test, n_outputs)
    Yhat_test : ndarray or None
        Predicted output data for testing, with shape (n_samples_test, n_outputs)
    fit : str, optional
        Metrics used: 'R2' (default), or 'BFR', or 'RMSE', or 'Accuracy'

    Returns
    -------
    score_train: array
        Score on training data (one entry per output). If training data are not provided, a NaN is returned.
    score_test : array
        Score on test data (one entry per output). If test data are not provided, a NaN is returned.
    msg : string
        Printout summarizing computed performance results
        """

    use_training = Y_train is not None and Yhat_train is not None
    use_test = Y_test is not None and Yhat_test is not None

    if use_training:
        Y_train = vec_reshape(Y_train)
        Yhat_train = vec_reshape(Yhat_train)
        if Y_train.shape != Yhat_train.shape:
            raise ValueError(
                "Inconsistent dimensions between reference and predicted training output data.")
        ny = Y_train.shape[1]

    if use_test:
        Y_test = vec_reshape(Y_test)
        Yhat_test = vec_reshape(Yhat_test)
        if Y_test.shape != Yhat_test.shape:
            raise ValueError(
                "Inconsistent dimensions between reference and predicted test output data.")
        # this could be a repetition of the previous assignment if use_training is True
        ny = Y_test.shape[1]

    score_train = np.zeros(ny)
    score_test = np.zeros(ny)
    msg = ''
    isR2 = fit.lower() == 'r2'
    isBFR = fit.lower() == 'bfr'
    isRMSE = fit.lower() == 'rmse'
    isAcc = fit.lower()[0:3] == 'acc'
    if not (isR2 or isBFR or isRMSE or isAcc):
        raise ValueError(
            "Invalid fit metric, only 'R2', 'BFR', and 'Accuracy' are supported.")

    if isAcc:
        unit = "%"
    else:
        unit = ""

    if ny > 1:
        for i in range(ny):
            if isR2 or isBFR:
                if use_training:
                    nY_train2 = np.sum(
                        (Y_train[:, i] - np.mean(Y_train[:, i]))**2)
                if use_test:
                    nY_test2 = np.sum(
                        (Y_test[:, i] - np.mean(Y_test[:, i]))**2)
            text = f"y{i+1}: "
            if isR2:
                if use_training:
                    score_train[i] = 100. * \
                        (1. - np.sum((Yhat_train[:, i] -
                                      Y_train[:, i]) ** 2) / nY_train2)
                if use_test:
                    score_test[i] = 100. * \
                        (1. - np.sum((Yhat_test[:, i] -
                                      Y_test[:, i]) ** 2) / nY_test2)
            elif isBFR:
                if use_training:
                    score_train[i] = 100. * (1. - np.linalg.norm(
                        Yhat_train[:, i] - Y_train[:, i]) / np.sqrt(nY_train2))
                if use_test:
                    score_test[i] = 100. * (1. - np.linalg.norm(
                        Yhat_test[:, i] - Y_test[:, i]) / np.sqrt(nY_test2))
            elif isRMSE:
                if use_training:
                    score_train[i] = np.sqrt((np.sum((Yhat_train[:, i] -
                                                      Y_train[:, i]) ** 2))/Y_train.shape[0])
                if use_test:
                    score_test[i] = np.sqrt(np.sum((Yhat_test[:, i] -
                                                    Y_test[:, i]) ** 2)/Y_test.shape[0])
            elif isAcc:
                if use_training:
                    score_train[i] = np.mean(
                        Yhat_train[:, i] == Y_train[:, i])*100
                if use_test:
                    score_test[i] = np.mean(
                        Yhat_test[:, i] == Y_test[:, i])*100
            if not use_training:
                score_train[i] = np.nan
            if not use_test:
                score_test[i] = np.nan

            text += f"{fit} score: training = {score_train[i]: 5.4f}{unit}"
            if use_test:
                text += f", test = {score_test[i]: 5.4f}{unit}"
            msg += '\n' + text
            # print(text)
        msg += "\n-----\nAverage "
    else:
        if isR2 or isBFR:
            if use_training:
                nY_train2 = np.sum((Y_train-np.mean(Y_train))**2)
            if use_test:
                nY_test2 = np.sum((Y_test - np.mean(Y_test))**2)
        if isR2:
            if use_training:
                score_train = 100. * \
                    (1. - np.sum((Yhat_train - Y_train) ** 2) / nY_train2)
            if use_test:
                score_test = 100. * \
                    (1. - np.sum((Yhat_test - Y_test) ** 2) / nY_test2)
        elif isBFR:
            if use_training:
                score_train = 100. * \
                    (1. - np.linalg.norm(Yhat_train-Y_train) / np.sqrt(nY_train2))
            if use_test:
                score_test = 100. * (1. - np.linalg.norm(Yhat_test -
                                                         Y_test) / np.sqrt(nY_test2))
        elif isRMSE:
            if use_training:
                score_train = np.sqrt(np.sum((Yhat_train -
                                      Y_train) ** 2)/Y_train.shape[0])
            if use_test:
                score_test = np.sqrt(np.sum((Yhat_test -
                                             Y_test) ** 2)/Y_test.shape[0])
        elif isAcc:
            if use_training:
                score_train = np.mean(Yhat_train == Y_train)*100
            if use_test:
                score_test = np.mean(Yhat_test == Y_test)*100
        if not use_training:
            score_train = np.nan
        if not use_test:
            score_test = np.nan

    msg += f"{fit} score:  training = {np.sum(score_train) / ny: 5.4f}{unit}"
    if use_test:
        msg += f", test = {np.sum(score_test) / ny: 5.4f}{unit}"
    return score_train, score_test, msg


def print_eigs(A, num_digits=4):
    """Print the eigenvalues of a given square matrix.

    (C) 2023 A. Bemporad

    Parameters
    ----------
    A : array
        Input matrix
    num_digits : int
        Number of decimal digits to print
    """
    print("Eigenvalues:")
    eigs = np.linalg.eig(A)[0]
    for i in range(A.shape[0]):
        print(f"%5.{num_digits}f" % np.real(eigs[i]), end="")
        im = np.imag(eigs[i])
        if not im == 0.0:
            print(f" + j%5.{num_digits}f" % im)
        else:
            print("")
    return


def unscale_model(A, B, C, D, ymean, ygain, umean, ugain):
    """ Unscale linear state-space model after training on scaled inputs and outputs.

    Given the scaled state-space matrices A,B,C,D, and the scaling factors ymean,ygain,umean,ugain,
    transforms the model to receive unscaled inputs and produce unscaled outputs:

        x(k+1) = Ax(k)+B(u(k) - umean)*ugain 
               = Ax(k)+B*diag(ugain)*(u(k) - umean)
         ys(k) = Cx(k)+D(u(k) - umean)*ugain 
               = (y(k) - ymean)*ygain
         y(k)  = ys(k)/ygain + ymean  
               = diag(1./ygain)*Cx(k)+diag(1./ygain)*D*diag(ugain)*(u(k) - umean)+ymean

    Parameters
    ----------
    A : ndarray
        A matrix
    B : ndarray
        B matrix
    C : ndarray
        C matrix
    D : ndarray
        D matrix
    ymean : array
        Mean value of the output
    ygain : array   
        Inverse of output's standard deviation
    umean : array
        Mean value of the input
    ugain : array
        Inverse of input's standard deviation

    Returns
    -------
    A : ndarray
        unscaled A matrix
    B : ndarray
        unscaled B matrix
    C : ndarray
        unscaled C matrix
    D : ndarray
        unscaled D matrix
    ymean : array
        offset = mean value of the output
    umean : array
        offset = mean value of the input
    """
    B = B*ugain
    C = (C.T/ygain).T
    D = D*ugain
    D = (D.T/ygain).T
    return A, B, C, D, ymean, umean


def IO2ss(Y, U, nx, M):
    """Compute a linear state-space realization from input/output data using least squares and SVD.

    Given and input/output dataset, compute the state-space matrices A, B, C, D of a state-space model of order nx by first obtaining a FIR model of order M, and then performing a singular value decomposition of the corresponding Hankel matrix.

    (C) 2024 A. Bemporad, May 20, 2024
    
    References:

    [1] S.Y. Kung, "A New Identification and Model Reduction Algorithm via Singular Value Decomposition," In 12th Asilomar Conference on Circuits, Systems and Computers, pages 705–714, 1978.

    [2] D.N. Miller, R.A. de Callafon, "Subspace Identification From Classical Realization Methods,"
    Proc. 15th IFAC Sympt. System Identification, St. Malo, France, 2009.

    Parameters
    ----------
    Y : ndarray
        Output data array of shape (n_samples, ny)
    U : ndarray
        Input data array of shape (n_samples, nu)
    nx : int
        Desired system order
    M : int
        Intermediate FIR model order

    Returns
    -------
    A : ndarray
        A matrix
    B : ndarray
        B matrix
    C : ndarray
        C matrix
    D : ndarray
        D matrix
    S : ndarray
        Singular values of SVD decomposition of Hankel matrix, useful to choose the model order nx
        by looking at the first most significant singular values.
    """

    Y = vec_reshape(Y)
    U = vec_reshape(U)
    N, ny = Y.shape
    nu = U.shape[1]

    # solve least squares problem to retrieve FIR coefficients:
    bb = Y[M:]
    AA = np.hstack([U[M-k:N-k, :] for k in range(M)])
    h = lstsq(AA, bb)[0].T

    n1 = M//2
    n2 = M-n1-1
    # get Hankel matrix H = Gamma @ Omega
    H = np.vstack([h[:, nu*i:nu*(i+n2)] for i in range(1, n1+1)])

    # perform Singular Value Decomposition
    U1, S, Vt = svd(H, full_matrices=False)
    Ur = U1[:, :nx]
    Vr = Vt[:nx, :]

    Sr_sqrt = np.sqrt(S[:nx])
    Gamma_d = (Ur/Sr_sqrt).T
    Omega_d = Vr.T/Sr_sqrt

    # shifted Hankel matrix H1=Gamma@A@Omega
    H1 = np.vstack([h[:, nu*i:nu*(i+n2)] for i in range(2, n1+2)])
    A = Gamma_d@H1@Omega_d
    B = (Vr[:,:nu].T*Sr_sqrt).T
    C = Ur[:ny,:]*Sr_sqrt
    D = h[:ny, :nu]
    return A, B, C, D, S
