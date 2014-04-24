__author__ = 'brandonkelly'

import numpy as np
from sklearn import cross_validation, metrics
from sklearn.decomposition import PCA
import multiprocessing
import copy
import matplotlib.pyplot as plt


class SupervisedPCABase(object):

    def __init__(self, regressor, max_components=None, n_components=1, whiten=True):
        """
        Base class for performing supervised principal component regression. This is useful for cases where the number
        of inputs (features) is greater than the number of data points.

        @param regressor: The object that will perform the regression. The following members must be defined for this
            object:

            regressor.fit(X, y) : Fits the regression model y = f(X).
            regressor.predict(X) : Compute the prediction y = f(X).
            regressor.coef_score_ : The score of each parameter, used for ranking the most important features when
                                    computing the reduced feature space. In general this will be the absolute value of
                                    the coefficient value divided by its standard error. Note that this should *not*
                                    include the intercept.

        @param max_components: Maximum number of components to search over. The default is p.
        @param n_components: The number of reduced data matrix PCA components to use in the regression.
        @param whiten: Remove differences in variance among the components, i.e., principal components will have unit
            variance
        """
        self.regressor = regressor
        self.max_components = max_components
        self.pca_object = PCA(n_components=n_components, whiten=whiten)
        self.n_components = n_components
        self.whiten = whiten
        self.n_reduced = 0
        self.sort_idx = np.zeros(1)

    def _compute_stnd_coefs(self, X, y):
        """
        Compute the standardized regression coefficients, up to a common scaling factor.

        @param X: The matrix of inputs, shape (n,p).
        @param y: The array of response values, size n.
        @return: The standardized regression coefficients, size p.
        """
        p = X.shape[1]
        scoefs = np.zeros(p)
        for j in xrange(p):
            thisX = X[:, j]
            self.regressor.fit(thisX[:, np.newaxis], y)
            scoefs[j] = self.regressor.coef_score_

        return scoefs

    def _get_reduced_features(self, X, coefs, pmax):
        """
        Return the data projected onto the first n_components principal components computed using the reduced feature
        space.

        @param X: The array of inputs, shape (n, p).
        @param coefs: The array of standardized coefficients, size p.
        @param pmax: The maximum number of features to use in the reduced feature space PCA.
        @return: The data projected onto the reduced feature space PCA, shape (n, self.n_components).
        """
        sort_idx = np.argsort(coefs)[::-1]
        sort_idx = sort_idx[:pmax]
        self.pca_object.fit(X[:, sort_idx])
        X_reduced = self.pca_object.transform(X[:, sort_idx])

        return X_reduced, sort_idx

    def fit(self, X, y, n_reduced):
        """
        Perform the regression using the first self.n_components principal components from the reduced feature space.
        Note that this will call self.regressor.fit(X,y) to perform the regression.

        @param X: The array of inputs, shape (n, p).
        @param y: The array of response values, size n.
        @param n_reduced: The number of features to use in the reduced feature space.
        """
        scoefs = self._compute_stnd_coefs(X, y)
        X_reduced, sort_idx = self._get_reduced_features(X, scoefs, n_reduced)
        self.sort_idx = sort_idx
        self.regressor.fit(X_reduced, y)

    def predict(self, X):
        """
        Predict the value y = f(X) based on the PCA using the reduced feature space, based on the most recent call to
        self.fit(X, y, n_reduced).

        @param X: The array of inputs, shape (n, p).
        @return: The predicted values of the response.
        """
        X_reduced = self.pca_object.transform(X[:, self.sort_idx])
        y_predict = self.regressor.predict(X_reduced)
        return y_predict


def launch_coef_scores(args):
    """
    Wrapper to compute the standardized scores of the regression coefficients, used when computing the number of
    features in the reduced parameter set.

    @param args: Tuple containing the instance of SupervisedPCABase, feature matrix and response array.
    @return: The standardzed scores of the coefficients.
    """
    spca, X, y = args
    scoefs = spca._compute_stnd_coefs(X, y)
    return scoefs


def compute_cv_prediction(args):
    """
    Internal method to get predictions based on supervised PCA regression for each cross-validation fold. Need this
    format in order to compute the predictions for the CV folds in parallel.
    """
    spca, X_train, y_train, X_test, n_reduced, scoef = args
    SPCA = SupervisedPCABase(copy.deepcopy(spca.regressor), spca.max_components, spca.n_components, spca.whiten)
    X_reduced, sort_idx = SPCA._get_reduced_features(X_train, scoef, n_reduced)
    SPCA.regressor.fit(X_reduced, y_train)
    X_test_reduced = SPCA.pca_object.transform(X_test[:, sort_idx])
    y_predict = SPCA.regressor.predict(X_test_reduced)
    return y_predict


class SupervisedPCA(SupervisedPCABase):
    def __init__(self, regressor, max_components=None, n_components=1, whiten=True, n_jobs=1):
        """
        Class for performing supervised principal component regression. This is useful for cases where the number of
        inputs (features) is greater than the number of data points.

        @param regressor: The object that will perform the regression. The following members must be defined for this
            object:

            regressor.fit(X, y) : Fits the regression model y = f(X).
            regressor.predict(X) : Compute the prediction y = f(X).
            regressor.coef_score_ : The score of each parameter, used for ranking the most important features when
                                    computing the reduced feature space. In general this will be the absolute value of
                                    the coefficient value divided by its standard error. Note that this should *not*
                                    include the intercept.

        @param max_components: Maximum number of components to search over. The default is p.
        @param n_components: The number of reduced data matrix PCA components to use in the regression.
        @param whiten: Remove differences in variance among the components, i.e., principal components will have unit
            variance
        @param n_jobs: The number of threads to use for parallel processing. If n_jobs = -1 then use maximum number
            available.
        """
        super(SupervisedPCA, self).__init__(regressor, max_components, n_components, whiten)
        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count()
        self.n_jobs = n_jobs

    def _compute_cv_prediction(self, args):
        """
        Internal method to get predictions based on supervised PCA regression for each cross-validation fold. Need this
        format in order to compute the predictions for the CV folds in parallel.
        """
        X_train, y_train, X_test, n_reduced, scoef = args
        SPCA = SupervisedPCABase(copy.deepcopy(self.regressor), self.max_components, self.n_components, self.whiten)
        X_reduced, sort_idx = SPCA._get_reduced_features(X_train, scoef, n_reduced)
        SPCA.regressor.fit(X_reduced, y_train)
        X_test_reduced = SPCA.pca_object.transform(X_test[:, sort_idx])
        y_predict = SPCA.regressor.predict(X_test_reduced)
        return y_predict

    def _launch_coef_scores(self, args):
        """
        Wrapper to compute the standardized scores of the regression coefficients, used when computing the number of
        features in the reduced parameter set.

        @param args: Tuple containing the feature matrix and response array.
        @return: The standardzed scores of the coefficients.
        """
        X, y = args
        scoefs = self._compute_stnd_coefs(X, y)
        return scoefs

    def choose_nreduced(self, X, y, lossfunc=None, cv=None, verbose=False, cvplot=False):
        """
        Choose the number of features to use in the reduced feature set by minimizing the cross-validation error.

        @param X: The feature matrix, shape (n,p)
        @param y: The vector of response values, size n.
        @param lossfunc: The loss function to use for the CV error, callable. The default is mean squared error.
        @param cv: Number of CV folds (if int), or cross-validation iterator.
        @param verbose: Print helpful information.
        @param cvplot: Plot the CV error as a function of the number features in the reduced feature set.
        @return: The number of features in the reduced feature set that minimized the CV error.
        """
        if self.n_jobs > 1:
            pool = multiprocessing.Pool(self.n_jobs)
            pool.map(int, range(self.n_jobs))  # Trick to "warm up" the Pool

        # setup cross-validation iterator
        if cv is None:
            K_folds = 8
        if isinstance(cv, int):
            K_folds = cv

        cv = cross_validation.KFold(y.size, n_folds=K_folds)

        if lossfunc is None:
            lossfunc = metrics.mean_squared_error

        if self.max_components is None:
            self.max_components = X.shape[1]

        if verbose:
            print 'Searching over', self.max_components, ' features to include in the reduced feature space.'
            print 'Computing univariate regression tests statistics for each feature...'

        # first compute coefficients scores
        sargs = []
        for train_idx, test_idx in cv:
            if self.n_jobs == 1:
                sargs.append((X[train_idx, :], y[train_idx]))
            else:
                sargs.append((self, X[train_idx, :], y[train_idx]))

        if self.n_jobs == 1:
            scoefs = map(self._launch_coef_scores, sargs)
        else:
            scoefs = pool.map(launch_coef_scores, sargs)

        # find optimal number of features to use in PCA on reduced feature set, do this by minimizing cross-validation
        # error on a grid.
        cverrors = np.zeros(self.max_components)

        if verbose:
            print 'Computing cross-validation errors on a grid of up to', self.max_components, 'features used in the', \
                'reduced feature space...'

        for k in xrange(self.max_components):
            cverror_args = []
            ytest = []
            fold_idx = 0
            for train_idx, test_idx in cv:
                if self.n_jobs == 1:
                    cverror_args.append((X[train_idx, :], y[train_idx], X[test_idx, :], k + 1, scoefs[fold_idx]))
                else:
                    cverror_args.append((self, X[train_idx, :], y[train_idx], X[test_idx, :], k + 1, scoefs[fold_idx]))
                ytest.append(y[test_idx])
                fold_idx += 1

            if self.n_jobs == 1:
                ypredictions = map(self._compute_cv_prediction, cverror_args)
            else:
                ypredictions = pool.map(compute_cv_prediction, cverror_args)

            cverror_k = 0.0
            for yt, yp in zip(ytest, ypredictions):
                cverror_k += lossfunc(yt, yp) / K_folds
            cverrors[k] = cverror_k

        if cvplot:
            plt.plot(np.arange(1, self.max_components + 1), cverrors)
            plt.xlabel('# of features in reduced set')
            plt.ylabel('CV Loss Function')
            plt.show()

        n_reduced = cverrors.argmin() + 1

        if verbose:
            print 'Selected', n_reduced, 'features to use in the reduced feature set.'

        return n_reduced