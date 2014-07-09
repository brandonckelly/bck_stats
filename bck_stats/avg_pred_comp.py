__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy import linalg


def distance_matrix(Xvals):
    covar = np.cov(Xvals, rowvar=0)
    covar_inv = linalg.inv(covar)
    Dmat = cdist(Xvals, Xvals, metric='mahalanobis', VI=covar_inv)

    return Dmat


def impact_single_theta(predict, theta, X, predictor, weights, predict_args=None):
    # first compute the matrix of model predictions:
    #   y_predict[i, j] = E(y|u_i, v_j, theta)
    ndata = len(X[predictor])
    condition_on = X.keys().remove(predictor)  # labels for the set of v predictors
    V = []
    for key in condition_on:
        V.append(X[key])
    V = np.column_stack(V)  # the inactive set of predictors, we condition on these when looking for changes in u
    u = X[predictor]  # the active predictor
    predictor_idx = X.keys().index(predictor)
    y_predict = np.zeros((ndata, ndata))
    for i in range(ndata):
        # make sure we keep the original ordering of the predictors
        X_i = np.insert(V, predictor_idx, u[i] * np.ones(ndata))
        y_predict[i] = predict(X_i, theta, *predict_args)

    # get matrix of signs of transitions
    transition_sign = np.zeros((ndata, ndata))
    for j in range(ndata):
        transition_sign[:, j] = np.sign(u - u[j])

    u1, u2 = np.meshgrid(u, u)
    transition_sign = np.sign(u1 - u2)
    impact_theta = np.sum(weights * (y_predict - y_predict.diagonal()) * transition_sign)

    return impact_theta


def impact(predict, theta, X, predictors=None, predict_args=None, nneighbors=None, nx=None, ntheta=None,
           mahalanobis_constant=1.0):

    if predictors is None:
        # calculate the impact for all the predictors
        predictors = X.keys()

    if nx is not None:
        # use only a subset of the data points
        subset_idx = np.random.permutation(len(X.values()[0]))[:nx]
        for key in X.keys():
            X[key] = X[key][subset_idx]
    else:
        nx = len(X.values()[0])
    if ntheta is not None:
        # use only a subset of the theta samples
        subset_idx = np.random.permutation(theta.shape[0])
        theta = theta[subset_idx]
    else:
        ntheta = theta.shape[0]
    if nneighbors is None:
        # use all of the neighbors when computing the weights
        nneighbors = len(X.values()[0])

    Xvals = np.column_stack(X.values())

    # first compute the distance matrix
    Dmat = distance_matrix(Xvals)
    weights0 = 1.0 / (1.0 + Dmat)

    # get the sets of nearest neighbors
    knn = NearestNeighbors(n_neighbors=nneighbors)
    knn.fit(Xvals)
    nn_idx = knn.kneighbors(Xvals, return_distance=False)

    weights = np.zeros_like(weights0)
    for i in range(weights.shape[0]):
        # data points outside of K nearest neighbors have weight of zero
        weights[nn_idx[i], i] = weights0[nn_idx[i]]

    weights /= weights.sum(axis=0)  # normalize weights to contribution to impact for each data point is the same

    impacts = dict()
    impact_sigmas = dict()
    for predictor in predictors:
        impact_theta = np.zeros(theta.shape)
        for s in range(ntheta):
            impact_theta[s] = impact_single_theta(predict, theta[s], X, predictor, weights, predict_args=predict_args)
        impacts[predictor] = np.mean(impact_theta)
        impact_sigmas[predictor] = np.std(impact_theta)

    return impacts, impact_sigmas