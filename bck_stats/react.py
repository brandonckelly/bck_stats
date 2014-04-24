__author__ = 'brandonkelly'

import numpy as np
from sklearn.isotonic import IsotonicRegression


class REACT(object):

    def __init__(self, basis='DCT', n_components=None, method='monotone'):
        try:
            basis.lower() in ['dct', 'manual']
        except ValueError:
            'Input basis must be either DCT or manual.'

        try:
            method.lower() in ['monotone', 'nss']
        except ValueError:
            'method must be either monotone or nss.'

        self.basis = basis
        self.nfreq = 1
        self.ncomp = 1
        self.n_components = n_components
        self.method = method
        self.coefs = np.zeros(1)
        self.shrinkage_factors = np.zeros(1)

    def fit(self, y, X=None, sigsqr=None):

        # check inputs
        if X is None:
            # build the discrete cosine basis
            if self.n_components is None:
                n_components = len(y)
            else:
                n_components = self.n_components
            X = self.build_dct(len(y), n_components)
            self.nfreq = len(y)
            self.ncomp = n_components
        else:
            if self.n_components is None:
                n_components = X.shape[1]
            else:
                n_components = self.n_components
            self.ncomp = n_components

        try:
            n_components <= len(y)
        except ValueError:
            'Number of components must be less than the length of y.'

        self.coefs = np.dot(X.T, y)

        if sigsqr is None:
            # estimate noise variance using first difference estimator
            sigsqr = np.sum((y[1:] - y[:-1]) ** 2) / (2.0 * (len(y) - 1))

        if self.method == 'monotone':
            # use monotone shrinkage on the basis coefficients
            self._set_shrinkage_factors(sigsqr)
        else:
            # use nested subset selection to choose the order of the basis expansion
            self._set_nss_order(sigsqr)

        self.coefs *= self.shrinkage_factors

        ysmooth = X.dot(self.coefs)
        return ysmooth

    @staticmethod
    def build_dct(n, p):
        rows, columns = np.mgrid[:n, :p]
        U = np.cos(np.pi * rows * columns / (n - 1.0))
        row_norm = 2 * np.ones(n)
        row_norm[0] = 1.0
        row_norm[-1] = 1.0
        col_norm = 2 * np.ones(p)
        col_norm[0] = 1.0
        if p == n:
            col_norm[-1] = 1.0
        U *= 0.5 * np.sqrt(2.0 * np.outer(row_norm, col_norm) / (n - 1))

        return U

    def interpolate(self, x_idx):
        try:
            self.method.lower() == 'dct'
        except AttributeError:
            'Interpolation only available for DCT basis.'

        n = self.nfreq
        p = self.ncomp
        cols = np.arange(p)
        row_norm = 2 * np.ones(n)
        row_norm[0] = 1.0
        row_norm[-1] = 1.0
        col_norm = 2 * np.ones(p)
        col_norm[0] = 1.0
        U = np.cos(np.pi * np.outer(x_idx / n, cols))
        U *= 0.5 * np.sqrt(2.0 * np.outer(row_norm, col_norm) / (n - 1))
        y_interp = U.dot(self.coefs)
        return y_interp

    def _set_shrinkage_factors(self, sigsqr):
        coefs_snr = (self.coefs ** 2 - sigsqr) / self.coefs ** 2  # signal-to-noise ratio of the coefficients
        coefs_snr[coefs_snr < 0] = 0.0
        x = np.arange(len(coefs_snr))
        weights = self.coefs ** 2
        self.shrinkage_factors = \
            IsotonicRegression(y_min=0.0, y_max=1.0, increasing=False).fit_transform(x, coefs_snr, weights)

    def _set_nss_order(self, sigsqr):
        coefs_snr = (self.coefs ** 2 - sigsqr) / self.coefs ** 2  # signal-to-noise ratio of the coefficients
        coefs_snr[coefs_snr < 0] = 0.0
        risk = np.empty(len(coefs_snr))
        shrinkage_factor = np.zeros(len(coefs_snr))
        for j in xrange(len(risk)):
            shrinkage_factor[:j+1] = 1.0
            risk[j] = np.mean((shrinkage_factor - coefs_snr) ** 2 * self.coefs ** 2)
        best_order = risk.argmin()
        self.shrinkage_factors = np.ones(len(coefs_snr))
        self.shrinkage_factors[best_order:] = 0.0  # only keep first best_order basis coefficients


class REACT2D(REACT):
    def __init__(self, max_order=None, method='monotone'):
        # currently only support the DCT for 2-D data
        super(REACT2D, self).__init__('DCT', max_order, method)
        self.row_order = np.zeros(1)
        self.col_order = np.zeros(1)

    def interpolate(self, x_idx):
        if True:
            print 'Interpolation not currently available for REACT2D'
        else:
            super(REACT2D, self).interpolate(x_idx)

    @staticmethod
    def build_dct(nrows, ncols, p):
        # first build 1-D basis vectors
        Urows = super(REACT2D, REACT2D).build_dct(nrows, p)
        Ucols = super(REACT2D, REACT2D).build_dct(ncols, p)
        # now build 2-d basis as outer products of 1-d basis vectors
        row_order, col_order = np.mgrid[:p, :p]
        row_order = row_order.ravel() + 1
        col_order = col_order.ravel() + 1
        # sort the basis images by the sum of squares of their orders
        sqr_order = row_order ** 2 + col_order ** 2
        s_idx = np.argsort(sqr_order)
        row_order = row_order[s_idx]
        col_order = col_order[s_idx]
        U = np.empty((nrows * ncols, len(row_order)))
        for j in xrange(len(row_order)):
            U[:, j] = np.outer(Urows[:, row_order[j]-1], Ucols[:, col_order[j]-1]).ravel()

        return U

    def fit(self, y, sigsqr):
        # build the discrete cosine basis
        if self.n_components is None:
            components_from_y = True
            self.n_components = min(y.shape)
        else:
            components_from_y = False

        try:
            self.n_components <= min(y.shape)
        except ValueError:
            'Number of components must be less than the length of y.'

        # build the 2-D DCT here and then feed into REACT.fit()
        X = self.build_dct(y.shape[0], y.shape[1], self.n_components)

        ysmooth = super(REACT2D, self).fit(y.ravel(), X, sigsqr)

        # save the orders of the basis functions
        row_order, col_order = np.mgrid[:self.n_components, :self.n_components]
        row_order = row_order.ravel() + 1
        col_order = col_order.ravel() + 1
        # sort the basis images by the sum of squares of their orders
        sqr_order = row_order ** 2 + col_order ** 2
        s_idx = np.argsort(sqr_order)
        self.row_order = row_order[s_idx]
        self.col_order = col_order[s_idx]

        if components_from_y:
            # return n_components to value from constructor
            self.n_components = None

        return np.reshape(ysmooth, y.shape)
