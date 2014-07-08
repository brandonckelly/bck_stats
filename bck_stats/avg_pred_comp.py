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


def impact_theta(predict, theta, )


def impact(predict, theta, X, predictors=None, predict_args=None, nneighbors=None, nx=None, ntheta=None,
           mahalanobis_constant=1.0):

    if predictors is None:
        # calculate the impact for all the predictors
        predictors = X.keys()

    if nx is None:
        # use all of the data points
        nx = len(X.values()[0])
    if ntheta is None:
        # use all of the theta samples
        ntheta = theta.shape[0]

    Xvals = np.column_stack(X.values())

    # first compute the distance matrix
    Dmat = distance_matrix(Xvals)
    weights = 1.0 / (1.0 + Dmat)

    # get the sets of nearest neighbors
    if nneighbors is not None:
        nn_idx = NearestNeighbors(n_neighbors=nneighbors).fit()
    else:
        # use all of the data points for the weights
        nn_idx = np.arange(len(X.values()[0]))

