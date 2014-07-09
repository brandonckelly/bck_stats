__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy import linalg
import multiprocessing


def distance_matrix(Xvals):
    covar = np.cov(Xvals, rowvar=0)
    covar_inv = linalg.inv(covar)
    Dmat = cdist(Xvals, Xvals, metric='mahalanobis', VI=covar_inv)

    return Dmat


def impact_single_theta(args):
    predict, theta, X, p_idx, weights, predict_args = args
    # first compute the matrix of model predictions:
    #   y_predict[i, j] = E(y|u_i, v_j, theta)
    ndata = X.shape[0]
    X_copy = X.copy()
    u = X[:, p_idx]  # the active predictor
    y_predict = np.zeros((ndata, ndata))
    for i in range(ndata):
        X_copy[:, p_idx] = u[i]
        y_predict[i] = predict(X_copy, theta, *predict_args)

    # get matrix of signs of transitions
    transition_sign = np.zeros((ndata, ndata))
    for j in range(ndata):
        transition_sign[:, j] = np.sign(u - u[j])

    u1, u2 = np.meshgrid(u, u)
    transition_sign = np.sign(u2 - u1)
    y_predict_diff = y_predict - np.outer(np.ones(ndata), y_predict.diagonal())
    numer = np.sum(weights * y_predict_diff * transition_sign)  # signed version
    abs_numer = np.sum(weights * np.abs(y_predict_diff))  # absolute version
    # denom = np.sum(weights * (u2 - u1) * np.sign(u2 - u1))
    denom = np.sum(weights)

    return numer / denom, abs_numer / denom


def impact(predict, theta, X, predictors=None, predict_args=None, nneighbors=None, nx=None, ntheta=None,
           mahalanobis_constant=1.0, n_jobs=1):

    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()
    if n_jobs > 1:
        pool = multiprocessing.Pool(n_jobs)

    if predictors is None:
        # calculate the impact for all the predictors
        predictors = np.arange(X.shape[1])

    if nx is not None:
        # use only a subset of the data points
        subset_idx = np.random.permutation(X.shape[0])[:nx]
        X = X[subset_idx]
    else:
        nx = X.shape[0]
    if ntheta is not None:
        # use only a subset of the theta samples
        subset_idx = np.random.permutation(theta.shape[0])
        theta = theta[subset_idx]
    else:
        ntheta = theta.shape[0]
    if nneighbors is None:
        # use all of the neighbors when computing the weights
        nneighbors = X.shape[0]

    # first compute the distance matrix
    Dmat = distance_matrix(X)
    weights0 = 1.0 / (mahalanobis_constant + Dmat)

    # get the sets of nearest neighbors
    knn = NearestNeighbors(n_neighbors=nneighbors)
    knn.fit(X)
    nn_idx = knn.kneighbors(X, return_distance=False)

    weights = np.zeros_like(weights0)
    for i in range(weights.shape[0]):
        # data points outside of K nearest neighbors have weight of zero
        weights[nn_idx[i], i] = weights0[nn_idx[i], i]

    weights /= weights.sum(axis=0)  # normalize weights to contribution to impact for each data point is the same

    impacts = np.zeros(len(predictors))
    abs_impacts = np.zeros_like(impacts)
    impact_sigmas = np.zeros_like(impacts)
    abs_impact_sigma = np.zeros_like(impacts)
    print 'Doing predictor'
    for p_idx in predictors:
        print p_idx, '...'
        args = []
        for s in range(ntheta):
            args.append([predict, theta[s], X, p_idx, weights, predict_args])
        if n_jobs == 1:
            results = map(impact_single_theta, args)
        else:
            results = pool.map(impact_single_theta, args)
        results = np.array(results)
        impacts[p_idx] = np.mean(results[:, 0])
        impact_sigmas[p_idx] = np.std(results[:, 0])
        abs_impacts[p_idx] = np.mean(results[:, 1])
        abs_impact_sigma[p_idx] = np.std(results[:, 1])

        # impact_theta = np.zeros(theta.shape)
        # impact_theta_abs = np.zeros_like(impact_theta)
        # for s in range(ntheta):
        #     impact_s, abs_impact_s = impact_single_theta(predict, theta[s], X, p_idx, weights, predict_args=predict_args)
        #     impact_theta[s] = impact_s
        #     impact_theta_abs[s] = abs_impact_s
        # impacts[p_idx] = np.mean(impact_theta)
        # impact_sigmas[p_idx] = np.std(impact_theta)
        # abs_impacts[p_idx] = np.mean(impact_theta_abs)
        # abs_impact_sigma[p_idx] = np.std(impact_theta_abs)

    return impacts, impact_sigmas, abs_impacts, abs_impact_sigma


if __name__ == "__main__":
    # test and example usage
    ndata = 200
    beta = np.array([1.0, 2.0, -0.6, 0.1])
    sigma = 0.1
    X = np.column_stack((np.ones(ndata), np.random.standard_normal(ndata), np.random.uniform(0.0, 5.0, ndata),
                         np.random.standard_cauchy(ndata)))
    y = X.dot(beta) + sigma * np.random.standard_normal(ndata)

    XX_inv = linalg.inv(X.T.dot(X))
    bhat = XX_inv.dot(X.T.dot(y))
    bcov = XX_inv * sigma * sigma

    nsamples = 100
    betas = np.random.multivariate_normal(bhat, bcov, nsamples)
    betas = betas[:, 1:]  # ignore constant term

    def linear_mean(X, beta, constant):
        ymean = X.dot(beta) + constant
        return ymean

    # don't include constant term
    impacts, isigmas, abs_impacts, aisigmas = \
        impact(linear_mean, betas, X[:, 1:], predict_args=(bhat[0],), nneighbors=20, n_jobs=4)
    print impacts
    sorted_idx = np.argsort(np.abs(impacts))

    labels = np.array(['x1', 'x2', 'x3'])[sorted_idx]

    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, impacts[sorted_idx], align='center', xerr=isigmas[sorted_idx], alpha=0.5)
    plt.yticks(pos, labels)
    plt.xlabel('Impact')
    plt.show()