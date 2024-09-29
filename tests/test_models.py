# jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

import jax
from jax_sysid.utils import standard_scale, unscale, compute_scores
import numpy as np
import jax.numpy as jnp
from jax_sysid.models import Model, LinearModel
import unittest

# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package
#
# Authors: A. Bemporad


jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations


class Test_jax_sysid(unittest.TestCase):

    def test_linear_vs_nonlinear(self):
        # Data generation
        seed = 3  # for reproducibility of results
        np.random.seed(seed)

        nx = 8  # number of states
        ny = 3  # number of outputs
        nu = 3  # number of inputs

        # number of epochs for Adam and L-BFGS-B optimization
        adam_epochs, lbfgs_epochs = 1000, 1000
        rho_th = 1.e-3  # L2-regularization on model coefficients
        rho_x0 = 1.e-2  # L2-regularization on initial state
        tau_th = 0*0.03  # L1-regularization on model coefficients
        tau_g = 0.*0.15  # group Lasso penalty
        # entries smaller than this are set to zero (only if tau_th>0 or tau_g>0)
        zero_coeff = 1.e-4

        N_train = 1000  # number of training data
        N_test = 1000  # number of test data
        Ts = 1.  # sample time

        # True linear dynamics
        At = np.random.randn(nx, nx)
        # makes matrix strictly Schur
        At = At/np.max(np.abs(np.linalg.eig(At)[0]))*0.95
        Bt = np.random.randn(nx, nu)
        Ct = np.random.randn(ny, nx)
        Dt = np.zeros((ny, nu))  # no direct feedthrough

        qy = 0.02  # output noise std
        qx = 0.02  # process noise std

        U_train = np.random.rand(N_train, nu)-0.5
        x0_train = np.random.randn(nx)
        # Create true model to generate the training dataset
        truemodel = LinearModel(nx, ny, nu, feedthrough=False)
        truemodel.init(params=[At, Bt, Ct], x0=x0_train)
        Y_train, X_train = truemodel.predict(x0_train, U_train, qx, qy)
        Ys_train, ymean, ygain = standard_scale(Y_train)
        Us_train, umean, ugain = standard_scale(U_train)

        U_test = np.random.rand(N_test, nu)-0.5
        x0_test = np.random.randn(nx)
        Y_test, X_test = truemodel.predict(x0_test, U_test, qx, qy)
        Ys_test = (Y_test-ymean)*ygain  # use same scaling as for training data
        Us_test = (U_test-umean)*ugain

        # make model a linear state-space model without direct feedthrough
        model1 = LinearModel(nx, ny, nu, feedthrough=False)

        @jax.jit
        def state_fcn(x, u, params):
            A, B, C = params
            return A@x+B@u

        @jax.jit
        def output_fcn(x, u, params):
            A, B, C = params
            return C@x
        params_init = [
            0.5*np.eye(nx), 0.1*np.random.randn(nx, nu), 0.1*np.random.randn(ny, nx)]
        model2 = Model(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn)
        model2.init(params=params_init)

        model1.loss(rho_x0=rho_x0, rho_th=rho_th, tau_th=tau_th,
                    tau_g=tau_g, zero_coeff=zero_coeff)
        model2.loss(rho_x0=rho_x0, rho_th=rho_th, tau_th=tau_th,
                    tau_g=tau_g, zero_coeff=zero_coeff)
        model1.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)
        model2.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs)

        model1.fit(Ys_train, Us_train)
        model2.fit(Ys_train, Us_train)

        Yshat_train1, _ = model1.predict(model1.x0, Us_train)
        Yhat_train1 = unscale(Yshat_train1, ymean, ygain)

        # use RTS Smoother to learn x0
        x0_test1 = model1.learn_x0(Us_test, Ys_test)
        Yshat_test1, _ = model1.predict(x0_test1, Us_test)
        Yhat_test1 = unscale(Yshat_test1, ymean, ygain)
        R21, R2_test1, msg1 = compute_scores(
            Y_train, Yhat_train1, Y_test, Yhat_test1, fit='R2')

        Yshat_train2, _ = model2.predict(model2.x0, Us_train)
        Yhat_train2 = unscale(Yshat_train2, ymean, ygain)

        # use RTS Smoother to learn x0
        x0_test2 = model2.learn_x0(Us_test, Ys_test)
        Yshat_test2, _ = model2.predict(x0_test2, Us_test)
        Yhat_test2 = unscale(Yshat_test2, ymean, ygain)
        R22, R2_test2, msg2 = compute_scores(
            Y_train, Yhat_train2, Y_test, Yhat_test2, fit='R2')

        print("Average R2 score on training data: linear model = %5.4f" %
              np.mean(R21), ", nonlinear model = %5.4f" % np.mean(R22))
        print("Average R2 score on test data: linear model = %5.4f" %
              np.mean(R2_test1), ", nonlinear model = %5.4f" % np.mean(R2_test2))

        self.assertEqual(np.maximum(np.abs(np.mean(
            R2_test1)-np.mean(R2_test2))-1.e-3, 0.0), 0.0, 'R2 test scores are different')
        self.assertEqual(np.maximum(np.abs(np.mean(R21)-np.mean(R22)) -
                         1.e-3, 0.0), 0.0, 'R2 test scores are different')
        return


if __name__ == '__main__':
    unittest.main()
