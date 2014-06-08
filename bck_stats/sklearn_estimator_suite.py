__author__ = 'brandonkelly'

import numpy as np
import abc

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, make_scorer, mean_absolute_error, mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.base import clone

float_types = (float, np.float, np.float32, np.float64, np.float_, np.float128, np.float16)
int_types = (int, np.int, np.int8, np.int16, np.int32, np.int64)


class GbcAutoNtrees(GradientBoostingClassifier):
    """
    Same as GradientBoostingClassifier, but the number of estimators is chosen automatically by maximizing the
    out-of-bag score.
    """
    def __init__(self, subsample, loss='deviance', learning_rate=0.01, n_estimators=500, min_samples_split=2,
                 min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=0):
        super(GbcAutoNtrees, self).__init__(loss, learning_rate, n_estimators, subsample, min_samples_split,
                                            min_samples_leaf, max_depth, init, random_state, max_features, verbose)

    def fit(self, X, y):

        super(GbcAutoNtrees, self).fit(X, y)
        oob_score = np.cumsum(self.oob_improvement_)
        ntrees = oob_score.argmax() + 1
        if self.verbose:
            print 'Chose', ntrees, 'based on the OOB score.'
        self.n_estimators = ntrees
        self.estimators_ = self.estimators_[:ntrees]

        # plt.plot(oob_score)
        # plt.show()

        return self


class GbrAutoNtrees(GradientBoostingRegressor):
    """
    Same as GradientBoostingRegressor, but the number of estimators is chosen automatically by maximizing the
    out-of-bag score.
    """

    def __init__(self, subsample, loss='ls', learning_rate=0.1, n_estimators=100, min_samples_split=2,
                 min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9,
                 verbose=0):
        super(GbrAutoNtrees, self).__init__(loss, learning_rate, n_estimators, subsample, min_samples_split,
                                            min_samples_leaf, max_depth, init, random_state, max_features, alpha,
                                            verbose)

    def fit(self, X, y):

        super(GbrAutoNtrees, self).fit(X, y)
        oob_score = np.cumsum(self.oob_improvement_)
        ntrees = oob_score.argmax() + 1
        if self.verbose:
            print 'Chose', ntrees, 'based on the OOB score.'
        self.n_estimators = ntrees
        self.estimators_ = self.estimators_[:ntrees]

        # plt.plot(oob_score)
        # plt.show()

        return self


class BasePredictorSuite(object):
    """ Base class for running a suite of estimators from scikit-learn. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, tuning_ranges=None, models=None, cv=None, njobs=1, pre_dispatch='2*n_jobs', stack=True,
                 verbose=False):
        """
        Initialize a pipeline to run a suite of scikit-learn estimators. The tuning parameters are chosen through
        cross-validation or the out-of-bags score (for Random Forests) as part of the fitting process.

        :param tuning_ranges: A nested dictionary containing the ranges of the tuning parameters. It should be of the
            format {model name 1: {parameter name 1: list(value range 1), parameter name 2: list(value range 2), ...} }.
        :param models: A list of instantiated scikit-learn estimator classes to fit. If None, these are taken from
            the models listed in tuning_range.
        :param cv: The number of CV folds to use, or a CV generator.
        :param njobs: The number of processes to run in parallel.
        :param pre_dispatch: Passed to sklearn.grid_search.GridSearchCV, see documentation for GridSearchCV for further
            details.
        :param stack: If true, then the predict() method will return a stacked (averaged) value over the estimators.
            Otherwise, if false, then predict() will return the predictions for each estimator.
        :param verbose: If true, print out helpful information.
        """
        super(BasePredictorSuite, self).__init__()
        self.verbose = verbose
        if tuning_ranges is None:
            tuning_ranges = dict()
        self.tuning_ranges = tuning_ranges
        if models is None:
            models = []
        self.models = models
        self.model_names = []
        for model in self.models:
            # store the names of the sklearn classes used
            self.model_names.append(model.__class__.__name__)
            try:
                # make sure the model names are in the dictionary of tuning parameters
                model.__class__.__name__ in tuning_ranges
            except ValueError:
                'Could not find tuning parameters for', model.__class__.__name__

        if cv is None:
            cv = 3
        self.cv = cv
        self.njobs = njobs
        self.pre_dispatch = pre_dispatch
        self.scorer = None
        self.stack = stack
        self.best_scores = dict()
        self.nfeatures = None

    def refine_grid(self, best_params, model_name):
        """
        Refine the tuning parameter grid to zoom in on the region near the current maximum.

        :param best_params: A dictionary containing the set of best tuning parameter names and their values. Should be
            of the form {'parameter 1': value1, 'parameter 2', value2, ... }. The tuning parameter grid will be refined
            in the region of these parameter values.
        :param model_name: The name of the estimator corresponding to the tuning parameters in best_params.
        """
        for param_name in best_params:
            pvalue_list = self.tuning_ranges[model_name][param_name]
            best_value = best_params[param_name]
            # find the values corresponding to
            idx = pvalue_list.index(best_value)
            ngrid = len(pvalue_list)
            if idx == 0:
                # first element of grid, so expand below it
                if type(pvalue_list[0]) in int_types:
                    pv_min = pvalue_list[0] / 2  # reduce minimum grid value by a factor of 2
                    pv_min = max(1, pv_min)  # assume integer tuning parameters are never less than 1.
                    pv_max = pvalue_list[1]
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.linspace(pv_min, pv_max, ngrid).astype(np.int)))
                else:
                    # use logarithmic grids for floats
                    dp = np.log10(pvalue_list[1]) - np.log10(pvalue_list[0])
                    pv_min = np.log10(pvalue_list[0]) - dp
                    pv_max = np.log10(pvalue_list[1])
                    self.tuning_ranges[model_name][param_name] = list(np.logspace(pv_min, pv_max, ngrid))
                    if self.verbose:
                        print self.tuning_ranges[model_name][param_name]
            elif idx == ngrid - 1:
                # last element of grid, so expand above it
                if pvalue_list[idx] is None:
                    # special situation for some estimators, like the DecisionTreeClassifier
                    pv_min = pvalue_list[idx-1]  # increase the maximum grid value by a factor of 2
                    pv_max = 2 * pv_min
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.linspace(pv_min, pv_max, ngrid-1).astype(np.int)))
                    # make sure we keep None as the last value in the list
                    self.tuning_ranges[model_name][param_name].append(None)
                elif type(pvalue_list[idx]) in int_types:
                    pv_min = np.log10(pvalue_list[idx-1])
                    pv_max = np.log10(2 * pvalue_list[idx])  # increase the maximum grid value by a factor of 2
                    if param_name == 'max_features':
                        # can't have max_features > nfeatures
                        pv_max = min(2 * pvalue_list[idx], self.nfeatures)
                        pv_max = np.log10(pv_max)
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.logspace(pv_min, pv_max, ngrid).astype(np.int)))
                else:
                    # use logarithmic grids for floats
                    dp = np.log10(pvalue_list[idx]) - np.log10(pvalue_list[idx-1])
                    pv_min = np.log10(pvalue_list[idx-1])
                    pv_max = np.log10(pvalue_list[idx]) + dp
                    self.tuning_ranges[model_name][param_name] = list(np.logspace(pv_min, pv_max, ngrid))
                    if self.verbose:
                        print self.tuning_ranges[model_name][param_name]
            else:
                # inner element of grid
                if pvalue_list[idx + 1] is None:
                    # special situation for some estimators, like the DecisionTreeClassifier
                    pv_min = pvalue_list[idx-1]  # increase the maximum grid value by a factor of 2
                    pv_max = 2 * pvalue_list[idx]
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.linspace(pv_min, pv_max, ngrid-1).astype(np.int)))
                    # make sure we keep None as the last value in the list
                    self.tuning_ranges[model_name][param_name].append(None)
                elif type(pvalue_list[idx]) in int_types:
                    pv_min = pvalue_list[idx-1]
                    pv_max = pvalue_list[idx+1]
                    # switch to linear spacing for interior integer grid values
                    self.tuning_ranges[model_name][param_name] = \
                        list(np.unique(np.linspace(pv_min, pv_max, ngrid).astype(np.int)))
                else:
                    # use logarithmic grids for floats
                    pv_min = np.log10(pvalue_list[idx-1])
                    pv_max = np.log10(pvalue_list[idx+1])
                    self.tuning_ranges[model_name][param_name] = list(np.logspace(pv_min, pv_max, ngrid))
                    if self.verbose:
                        print self.tuning_ranges[model_name][param_name]

            # print 'New Grid:', self.tuning_ranges[model_name][param_name]

    def cross_validate(self, X, model_idx, y):
        """
        Fit the tuning parameters for an estimator on a grid using cross-validation.

        :param X: The array of predictors, shape (n_samples, n_features).
        :param model_idx: The index of the estimator to fit.
        :param y: The array of response values, shape (n_samples) or (n_samples, n_outputs) depending on the estimator.
        :return: A tuple containing the scikit-learn estimator object with the best tuning parameters, the score
            corresponding to the best tuning parameters, and a dictionary containing the best tuning parameter values.
        """
        if self.verbose:
            print 'Doing cross-validation for model', self.model_names[model_idx], '...'
        grid = GridSearchCV(self.models[model_idx], self.tuning_ranges[self.model_names[model_idx]],
                            scoring=self.scorer, n_jobs=self.njobs, cv=self.cv, pre_dispatch=self.pre_dispatch)
        grid.fit(X, y)
        if self.verbose:
            print 'Best', self.model_names[model_idx], 'has:'
            for tuning_parameter in self.tuning_ranges[self.model_names[model_idx]]:
                print '    ', tuning_parameter, '=', grid.best_params_[tuning_parameter]
            print '     CV Score of', grid.best_score_
        return grid.best_estimator_, grid.best_score_, grid.best_params_

    def oob_validate(self, X, model_idx, y):
        """
        Fit the tuning parameters for a Random Forest estimator on a grid by maximizing the score of the out-of-bag
        samples. This is faster than using cross-validation.

        :param X: The array of predictors, shape (n_samples, n_features).
        :param model_idx: The index of the estimator to fit.
        :param y: The array of response values, shape (n_samples) or (n_samples, n_outputs) depending on the estimator.
        :return: A tuple containing the scikit-learn estimator object with the best tuning parameters, the score
            corresponding to the best tuning parameters, and a dictionary containing the best tuning parameter values.
        """
        if self.verbose:
            print 'Doing OOB-validation for model', self.model_names[model_idx], '...'

        tune_grid = list(ParameterGrid(self.tuning_ranges[self.model_names[model_idx]]))

        best_estimator = None
        best_score = -1e30

        # fit random forest
        for point in tune_grid:
            estimator = clone(self.models[model_idx])
            for tpar in point:
                # set the tuning parameters
                estimator.__setattr__(tpar, point[tpar])
            estimator.fit(X, y)

            if estimator.oob_score_ > best_score:
                # new best values, save them
                best_score = estimator.oob_score_
                best_estimator = estimator
                best_params = estimator.get_params()

        best_tparams = dict()
        for tpar in self.tuning_ranges[self.model_names[model_idx]]:
            best_tparams[tpar] = best_params[tpar]  # only grab the values of the best tuning parameter

        if self.verbose:
            print 'Best', self.model_names[model_idx], 'has:'
            for tuning_parameter in self.tuning_ranges[self.model_names[model_idx]]:
                print '    ', tuning_parameter, '=', best_tparams[tuning_parameter]
            print '     OOB Score of', best_score

        return best_estimator, best_score, best_tparams

    def fit(self, X, y, n_refinements=1):
        """
        Fit the suite of estimators. The tuning parameters are estimated using cross-validation.

        :param X: The array of predictors, shape (n_samples, n_features).
        :param y: The array of response values, shape (n_samples) or (n_samples, n_outputs), depending on the estimator.
        :param n_refinements: The number of time to refine the grid of tuning parameter values. Must be an integer or
            dictionary. If an integer, the grid for all models will be refined this many times. If a dictionary, should
            have (key value) pairs given by (estimator name, n_refinements).
        :return: Returns self.
        """
        self.nfeatures = X.shape[1]
        ndata = len(y)
        try:
            X.shape[0] == ndata
        except ValueError:
            'X and y must have same number of rows.'

        if np.isscalar(n_refinements):
            # use same number of refinements for all models
            n_refinements = {name: n_refinements for name in self.model_names}

        if type(self.cv) in int_types:
            # construct cross-validation iterator
            self.cv = KFold(ndata, n_folds=self.cv)
        elif self.cv.n != ndata:
            # need to reconstruct cross-validation iterator since we have different data
            self.cv = KFold(ndata, n_folds=self.cv.n_folds)

        for k in range(len(self.models)):
            if 'RandomForest' in self.model_names[k]:
                # use out-of-bag error for validation error
                best_estimator, best_score, best_params = self.oob_validate(X, k, y)
            else:
                # use cross-validation for validation error
                best_estimator, best_score, best_params = self.cross_validate(X, k, y)

            self.models[k] = best_estimator
            self.best_scores[self.model_names[k]] = best_score

            for i in range(n_refinements[self.model_names[k]]):
                if self.verbose:
                    print 'Refining Grid...'
                old_score = best_score
                # now refine the grid and refit
                self.refine_grid(best_params, self.model_names[k])

                if 'RandomForest' in self.model_names[k]:
                    # use out-of-bag error for validation error
                    best_estimator, best_score, best_params = self.oob_validate(X, k, y)
                else:
                    # use cross-validation for validation error
                    best_estimator, best_score, best_params = self.cross_validate(X, k, y)
                if self.verbose:
                    print '     New Validation Score of', best_score, 'is an improvement of', \
                        100.0 * (best_score - old_score) / np.abs(old_score), '%.'

                self.models[k] = best_estimator
                self.best_scores[self.model_names[k]] = best_score

        return self

    def predict_all(self, X):
        """
        Predict the outputs as a function of the inputs for each model.

        :param X: The array of predictor values, shape (n_samples, n_features).
        :return: A dictionary containing the values of the response predicted at the input values for each model.
        """
        y_predict_all = {name: model.predict(X) for name, model in zip(self.model_names, self.models)}

        return y_predict_all

    @abc.abstractmethod
    def predict(self, X, weights='auto'):
        return self.predict_all(X)


class ClassificationSuite(BasePredictorSuite):

    def __init__(self, n_features=None, tuning_ranges=None, models=None, cv=None, njobs=1, pre_dispatch='2*n_jobs',
                 stack=True, verbose=False):
        """
        Initialize a pipeline to run a suite of scikit-learn classifiers. The tuning parameters are chosen through
        cross-validation or the out-of-bags score (for Random Forests) as part of the fitting process. The score
        function used is the accuracy score (fraction of correct classifications).

        :param verbose: Provide helpful output.
        :param n_features: The number of features that will be used when performing the fit. Must supply either
            n_features or tuning_ranges. This is necessary because the tuning parameter for the RandomForestClassifier
            is max_features, and max_features must be less than the number of features in the input array. So, in order
            to automatically construct the tuning_ranges dictionary it is necessary to know n_features in order to
            ensure max_features <= n_features.
        :param tuning_ranges: A nested dictionary containing the ranges of the tuning parameters. It should be of the
            format {model name 1: {parameter name 1: list(value range 1), parameter name 2: list(value range 2), ...} }.
            If n_features is not supplied, then tuning_ranges must be provided.
        :param models: A list of instantiated scikit-learn classifier classes to fit. If None, these are taken from
            the models listed in tuning_range.
        :param cv: The number of CV folds to use, or a CV generator.
        :param njobs: The number of processes to run in parallel.
        :param pre_dispatch: Passed to sklearn.grid_search.GridSearchCV, see documentation for GridSearchCV for further
            details.
        :param stack: If true, then the predict() method will return a stacked (averaged) value over the estimators.
            Otherwise, if false, then predict() will return the predictions for each estimator.
        """
        if tuning_ranges is None:
            try:
                n_features is not None
            except ValueError:
                'Must supply one of n_features or tuning_ranges.'
            # use default values for grid search over tuning parameters for all models
            tuning_ranges = {'LogisticRegression': {'C': list(np.logspace(-2.0, 1.0, 5))},
                             'DecisionTreeClassifier': {'max_depth': [5, 10, 20, 50, None]},
                             'LinearSVC': {'C': list(np.logspace(-2.0, 1.0, 5))},
                             'SVC': {'C': list(np.logspace(-2.0, 1.0, 5)),
                                     'gamma': list(np.logspace(np.log10(1.0 / n_features),
                                                               np.log10(1000.0 / n_features), 5))},
                             'RandomForestClassifier': {'max_features':
                                                        list(np.unique(np.linspace(2, n_features, 5).astype(np.int)))},
                             'GbcAutoNtrees': {'max_depth': [1, 2, 3, 5, 10]}}
        if models is None:
            # initialize the list of sklearn objects corresponding to different statistical models
            models = []
            if 'LogisticRegression' in tuning_ranges:
                models.append(LogisticRegression(penalty='l1', class_weight='auto'))
            if 'DecisionTreeClassifier' in tuning_ranges:
                models.append(DecisionTreeClassifier())
            if 'LinearSVC' in tuning_ranges:
                models.append(LinearSVC(penalty='l1', loss='l2', dual=False, class_weight='auto'))
            if 'SVC' in tuning_ranges:
                models.append(SVC(class_weight='auto'))
            if 'RandomForestClassifier' in tuning_ranges:
                models.append(RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=njobs))
            if 'GbcAutoNtrees' in tuning_ranges:
                models.append(GbcAutoNtrees(subsample=0.75, n_estimators=500, learning_rate=0.01))

        super(ClassificationSuite, self).__init__(tuning_ranges=tuning_ranges, models=models, cv=cv, njobs=njobs,
                                                  pre_dispatch=pre_dispatch, stack=stack, verbose=verbose)

        self.scorer = make_scorer(accuracy_score)
        self.nfeatures = n_features
        self.classes = None

    def predict(self, X, weights='auto'):
        """
        Predict the classes as a function of the inputs. If self.stack is true, then the predictions for each data point
        are computed based on a weighted majority vote of the estimators. Otherwise, a dictionary containing the
        predictions for each estimator are returns.

        :param X: The array of predictor values, shape (n_samples, n_features).
        :param weights: The weights to use when combining the predictions for the individual estimators. If 'auto', then
            the weights are given by the validation scores. If 'uniform', then uniform weights are used. Otherwise
            weights must be a dictionary with (model name, weight) as the (key, value) pair.
            No effect if self.stack = False.
        :return: The values of the response predicted at the input values.
        """
        y_predict_all = super(ClassificationSuite, self).predict_all(X)

        if weights is 'uniform':
            # just use uniform weighting
            weights = {name: 1.0 for name in self.model_names}

        if weights is 'auto':
            # weight based on validation score
            weights = self.best_scores

        if self.stack:
            # combine the model outputs
            y_votes = np.zeros((X.shape[0], len(self.model_names)))
            for name in y_predict_all:
                vote = y_predict_all[name]
                idx_1d = vote + np.arange(len(vote)) * y_votes.shape[1]
                # compute weighted vote for each class
                y_votes[np.unravel_index(idx_1d, y_votes.shape)] += weights[name]

            y_predict = self.classes[y_votes.argmax(axis=1)]  # output is winner of majority vote

        else:
            y_predict = y_predict_all

        return y_predict

    def fit(self, X, y, n_refinements=1):
        classes, y = np.unique(y, return_inverse=True)
        self.classes = classes
        return super(ClassificationSuite, self).fit(X, y, n_refinements)


class RegressionSuite(BasePredictorSuite):

    def __init__(self, n_features=None, tuning_ranges=None, models=None, cv=None, njobs=1, pre_dispatch='2*n_jobs',
                 stack=True, verbose=False, metric='lad'):
        try:
            metric.lower() in ['lad', 'mse']
        except ValueError:
            'Metric must be either lad or mse.'

        if tuning_ranges is None:
            try:
                n_features is not None
            except ValueError:
                'Must supply one of n_features or tuning_ranges.'
            # use default values for grid search over tuning parameters for all models
            tuning_ranges = {'DecisionTreeClassifier': {'max_depth': [5, 10, 20, 50, None]},
                             'RandomForestRegressor': {'max_features':
                                                       list(np.unique(np.linspace(2, n_features, 5).astype(np.int)))},
                             'GbrAutoNtrees': {'max_depth': [1, 2, 3, 5, 10]}}
        if models is None:
            # initialize the list of sklearn objects corresponding to different statistical models
            models = []
            if 'DecisionTreeRegressor' in tuning_ranges:
                models.append(DecisionTreeRegressor())
            if 'RandomForestRegressor' in tuning_ranges:
                models.append(RandomForestRegressor(n_estimators=500, oob_score=True, n_jobs=njobs))
            if 'GbrAutoNtrees' in tuning_ranges:
                models.append(GbrAutoNtrees(subsample=0.75, n_estimators=500, learning_rate=0.01))

        super(RegressionSuite, self).__init__(tuning_ranges, models, cv, njobs, pre_dispatch, stack, verbose)

        self.scorer = make_scorer(accuracy_score)
        self.nfeatures = n_features
        self.metric = metric.lower()
        if self.metric == 'lad':
            self.scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        elif self.metric == 'mse':
            self.scorer = make_scorer(mean_squared_error, greater_is_better=False)

    def predict(self, X, weights='auto'):

        y_predict_all = super(RegressionSuite, self).predict_all(X)

        if weights is 'uniform':
            # just use uniform weighting
            weights = {name: 1.0 for name in self.model_names}

        if weights is 'auto':
            # weight based on validation score
            weights = self.best_scores

        if self.stack:
            # combine the model outputs
            y_predict = 0.0
            wsum = 0.0
            for name in y_predict_all:
                y_predict += weights[name] * y_predict_all[name]
                wsum += weights[name]
            y_predict /= wsum
        else:
            y_predict = y_predict_all

        return y_predict