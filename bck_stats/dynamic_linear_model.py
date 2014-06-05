__author__ = 'brandonkelly'

import numpy as np
import pykalman
import matplotlib.pyplot as plt
import multiprocessing


def mae_loss(y, yfit):
    return np.mean(np.abs(y - yfit))


def _train_predict_dlm(args):
    """
    Helper function to train and predict the dynamic linear model for a train and test set. Seperated from the main
    class to enable the use of the multiprocessing module. This should not be called directly.
    """
    delta, X, y, ntrain, loss = args
    print delta
    dlm = DynamicLinearModel(include_constant=False)

    # first fit using the training data
    dlm.fit(X[:ntrain], y[:ntrain], delta=delta, method='filter')

    # now run the filter on the whole data set
    ntime, pfeat = X.shape
    observation_matrix = X.reshape((ntime, 1, pfeat))
    k = dlm.kalman
    kalman = pykalman.KalmanFilter(transition_matrices=k.transition_matrices,
                                   observation_matrices=observation_matrix,
                                   observation_offsets=k.observation_offsets,
                                   transition_offsets=k.transition_offsets,
                                   observation_covariance=k.observation_covariance,
                                   transition_covariance=k.transition_covariance,
                                   initial_state_mean=k.initial_state_mean,
                                   initial_state_covariance=k.initial_state_covariance)

    beta, bcov = kalman.filter(y)

    # predict the y-values in the test set
    yfit = np.sum(beta[ntrain-1:-1] * X[ntrain-1:-1], axis=1)

    test_error = loss(y[ntrain:], yfit)

    return test_error


class DynamicLinearModel(object):
    def __init__(self, include_constant=True):
        """
        Constructor for linear regression model with dynamic coefficients.
        """
        self.delta_grid = np.zeros(10)
        self.test_grid = np.zeros(10)
        self.delta = 1e-4
        self.test_error_ = 1.0
        self.kalman = pykalman.KalmanFilter()
        self.beta = np.zeros(2)
        self.beta_cov = np.identity(2)
        self.current_beta = np.zeros(2)
        self.current_bcov = np.identity(2)
        self.include_constant = include_constant

    @staticmethod
    def add_constant_(X):
        """
        Add a constant to the linear model by prepending a column of ones to the feature array.

        @param X: The feature array. Note that it will be overwritten, and the overwritten array will be returned.
        """
        if X.ndim == 1:
            # treat vector-valued X differently
            X = np.insert(X[:, np.newaxis], 0, np.ones(len(X)), axis=1)
        else:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

        return X

    def fit(self, X, y, method='smoother', delta=None, include_constant=None):
        """
        Fit the coefficients for the dynamic linear model.

        @param method: The method used to estimate the dynamic coefficients, either 'smoother' or 'filter'. If
            'smoother', then the Kalman Smoother is used, otherwise the Kalman Filter will be used. The two differ
             in the fact that the Kalman Smoother uses both future and past data, while the Kalman Filter only uses
             past data.
        @param X: The time-varying covariates, and (ntime, pfeat) array.
        @param y: The time-varying response, a 1-D array with ntime elements.
        @param delta: The regularization parameters on the time variation of the coefficients. Default is
            self.delta.
        @param include_constant: Boolean, if true then include a constant in the regression model.
        """
        try:
            method.lower() in ['smoother', 'filter']
        except ValueError:
            "method must be either 'smoother' or 'filter'."

        if delta is None:
            delta = self.delta
        else:
            self.delta = delta

        if include_constant is None:
            include_constant = self.include_constant
        else:
            self.include_constant = include_constant

        if include_constant:
            Xtemp = self.add_constant_(X.copy())
        else:
            Xtemp = X.copy()

        ntime, pfeat = Xtemp.shape

        observation_matrix = Xtemp.reshape((ntime, 1, pfeat))
        observation_offset = np.array([0.0])

        transition_matrix = np.identity(pfeat)
        transition_offset = np.zeros(pfeat)

        mu = (1.0 - delta) / delta
        # Var(beta_t - beta_{t-1}) = 1.0 / mu
        transition_covariance = np.identity(pfeat) / mu

        # parameters to be estimated using MLE
        em_vars = ['initial_state_mean', 'initial_state_covariance']
        kalman = pykalman.KalmanFilter(transition_matrices=transition_matrix, em_vars=em_vars,
                                       observation_matrices=observation_matrix,
                                       observation_offsets=observation_offset, transition_offsets=transition_offset,
                                       observation_covariance=np.array([1.0]),
                                       transition_covariance=transition_covariance)

        kalman.em(y)
        if method is 'smoother':
            beta, beta_covar = kalman.smooth(y)
        else:
            beta, beta_covar = kalman.filter(y)

        self.beta = beta
        self.beta_cov = beta_covar
        self.current_beta = beta[-1]
        self.current_bcov = beta_covar[-1]
        self.kalman = kalman

    def update(self, y, x):
        """
        Update the linear regression coefficients given the new values of the response and features.

        @param y: The new response value, a scalar.
        @param x: The new feature vector.
        """
        if self.include_constant:
            observation_matrix = np.insert(x, 0, 1.0)
        else:
            observation_matrix = x.copy()

        pfeat = observation_matrix.size
        observation_matrix = observation_matrix.reshape((1, pfeat))

        self.current_beta, self.current_bcov = \
            self.kalman.filter_update(self.current_beta, self.current_bcov, observation=y,
                                      observation_matrix=observation_matrix)

        self.beta = np.vstack((self.beta, self.current_beta))
        self.beta_cov = np.dstack((self.beta_cov.T, self.current_bcov)).T

    def predict(self, x):
        """
        Predict a value of the response given the input feature array and the current value of the coefficients.

        @param x: The input feature array.
        """
        if self.include_constant:
            xpredict = np.insert(x, 0, 1.0)
        else:
            xpredict = x

        return np.sum(self.current_beta * xpredict)

    def choose_delta(self, X, y, test_fraction=0.5, verbose=False, ndeltas=20, include_constant=True, loss=mae_loss,
                     njobs=1):
        """
        Choose the optimal regularization parameters for the linear smoother coefficients by minimizing an input loss
        function on a test set.

        @param X: The time-varying covariates, and (ntime, pfeat) array.
        @param y: The training set, a 1-D array.
        @param ndeltas: The number of grid points to use for the regularization parameter.
        @param test_fraction: The fraction of the input data to use as the test set, default is half.
        @param verbose: If true, then print the chosen regularization parameter and test error.
        @param include_constant: Boolean, include a constant in the linear model?
        @param loss: The loss function to use for evaluating the test error when choosing the regularization parameter.
            Must be of the form result = loss(ytest, yfit).
        @param njobs: The number of processors to use when doing the search over delta. If njobs = -1, all processors
            will be used.
        """

        if include_constant is None:
            include_constant = self.include_constant
        else:
            self.include_constant = include_constant

        if njobs < 0:
            njobs = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(njobs)
        pool.map(int, range(njobs))  # warm up the pool

        # split y into training and test sets
        ntime = y.size
        ntest = int(ntime * test_fraction)
        ntrain = ntime - ntest
        if X.ndim == 1:
            XX = X.reshape((X.size, 1))
        else:
            XX = X.copy()

        if include_constant:
            # add column of ones to feature array
            XX = self.add_constant_(XX)

        # grid of delta (regularization) values, between 1e-4 and 1.0.
        delta_grid = np.logspace(-4.0, np.log10(0.95), ndeltas)

        args = []
        for d in xrange(ndeltas):
            args.append((delta_grid[d], XX, y, ntrain, loss))

        if verbose:
            print 'Computing test errors...'

        if njobs == 1:
            test_grid = map(_train_predict_dlm, args)
        else:
            test_grid = pool.map(_train_predict_dlm, args)

        test_grid = np.array(test_grid)
        self.delta = delta_grid[test_grid.argmin()]
        self.test_error_ = test_grid.min()

        if verbose:
            print 'Best delta is', self.delta, 'and has a test error of', test_grid.min()

        self.delta_grid = delta_grid
        self.test_grid = test_grid


if __name__ == "__main__":
    # run test from Montana et al. (2009)
    nx = 1000
    x = np.zeros(nx)
    x[0] = np.random.uniform(-2.0, 2.0)
    for i in xrange(1, nx):
        x[i] = 0.8 * x[i-1] + np.random.uniform(-2.0, 2.0)

    y = np.zeros(x.size)
    beta = np.zeros(x.size)
    beta[0] = 2.0
    for i in xrange(1, x.size):
        if i < 300:
            beta[i] = beta[i-1] + 0.1 * np.random.standard_normal()
        elif i == 300:
            beta[i] = beta[i-1] + 4.0
        elif (i > 300) and (i < 600):
            beta[i] = beta[i-1] + 0.001 * np.random.standard_normal()
        else:
            beta[i] = 5.0 * np.sin(i / 10.0) + np.random.uniform(-2.0, 2.0)

    y = 2.0 + beta * x + 2.0 * np.random.standard_normal(nx)

    plt.plot(beta)
    plt.ylabel(r'$\beta$')
    plt.show()
    plt.clf()

    plt.plot(x, y, '.')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
    plt.clf()

    plt.plot(y)
    plt.ylabel('y')
    plt.show()
    plt.clf()

    dynamic = DynamicLinearModel(include_constant=False)
    dynamic.choose_delta(np.ones(len(y)), y, test_fraction=0.5, verbose=True, ndeltas=20, njobs=5)
    dynamic.fit(np.ones(len(y)), y)

    plt.semilogx(dynamic.delta_grid, dynamic.test_grid)
    plt.xlabel('Regularization (delta)')
    plt.ylabel('Mean Absolute Test Error')
    plt.show()

    plt.clf()
    for i in xrange(1):
        plt.subplot(2, 1, i + 1)
        plt.plot(y, '.')
        plt.plot(dynamic.beta[:, i])
        plt.ylabel(r"$\beta_" + str(i) + '$')
        if i == 1:
            plt.plot(beta, 'k')
    plt.show()