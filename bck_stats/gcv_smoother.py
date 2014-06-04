__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt


class GcvExpSmoother(object):
    def __init__(self, lookback=30):
        """
        Constructor for class to perform exponentially-weighted average smoothing of a 1-D data set.

        @param lookback: The maximum look-back length to use in the smoothing. Only the data points in
            y[idx - lookback:idx] are used to compute the smoothed estimate of y[idx+1].
        """
        self.lookback = int(lookback)  # support of exponential smoother, only use this many data points in computation
        self.efold = 1.0
        self.gcv_grid = np.zeros(2.0 * self.lookback)
        self.efold_grid = np.zeros(2.0 * self.lookback)

    def smooth(self, y):
        """
        Return a smoothed estimate of y, using the current value of self.efold for the e-folding length.

        @param y: The data, a 1-D array.
        @return: The smoothed estimate of y, a 1-D numpy array.
        """
        ysmooth, peff = self._smooth(self.efold, y)
        return ysmooth

    def weights(self, efold, lookback=None):
        if lookback is None:
            lookback = self.lookback
        xvalues = np.arange(0.0, lookback)
        weights = np.exp(-xvalues / efold)
        return weights[::-1] / np.sum(weights)

    def choose_efold(self, y, verbose=False):
        """
        Choose the optimal e-folding length of the exponential smoothing kernel using generalized cross-validation.

        @param y: The training set, a 1-D array.
        @param verbose: If true, then print the chosen smoothing length.
        """
        ngrid = 20
        efold_grid = np.logspace(-1.0, np.log10(self.lookback * 2.0), ngrid)
        gcv_grid = np.zeros(efold_grid.size)
        for i in xrange(efold_grid.size):
            smoothed_y, peffective = self._smooth(efold_grid[i], y)
            gcv_grid[i] = gcv_error(y, smoothed_y, peffective)

        # choose e-folding length of smoother to minimize the generalized cross-validation error
        self.efold = efold_grid[gcv_grid.argmin()]
        if verbose:
            print 'E-folding length chosen to be', self.efold

        # save the grids
        self.efold_grid = efold_grid
        self.gcv_grid = gcv_grid

    def _smooth(self, efold, y):
        try:
            y.size > self.lookback
        except ValueError:
            'Y must have at least self.lookback elements.'

        ysmooth = np.zeros(y.size)
        ysmooth[0] = y[0]

        peffective = 0.0  # trace of the smoothing matrix, the effective number of parameters

        # treat the first self.lookback data points seperately, since the base-line is shorter
        for i in xrange(1, self.lookback):
            weights = self.weights(efold, lookback=i)
            ysmooth[i] = weights.dot(y[0:i])
            peffective += weights[-1]

        weights = self.weights(efold)
        for i in xrange(y.size - self.lookback - 1):
            idx = self.lookback + i
            # estimate current y as exponentially-weighted average of previous self.lookback y-values
            ysmooth[idx] = weights.dot(y[idx - self.lookback:idx])
            peffective += weights[-1]

        ysmooth[-1] = weights.dot(y[y.size - self.lookback - 1:-1])
        peffective += weights[-1]

        return ysmooth, peffective


def gcv_error(y, ysmooth, peffective):
    """
    Compute generalized cross-validation error.

    @param y: The numpy array of y-values.
    @param ysmooth: The smoothed numpy array of y-values.
    @param peffective: The effective number of parameters of the smoothing matrix, given by its trace.
    @return: The generalized cross-validation error (L2-loss function).
    """
    gcv = np.mean((y - ysmooth) ** 2) / (1.0 - peffective / y.size) ** 2
    return gcv


if __name__ == "__main__":
    # example usage
    x = np.arange(500)
    y = np.cos(x / 15.0) + 0.1 * np.random.standard_normal(500)

    gcv = GcvExpSmoother()
    gcv.choose_efold(y, verbose=True)
    ysmooth = gcv.smooth(y)

    plt.semilogy(gcv.efold_grid, gcv.gcv_grid)
    plt.xlabel('E-folding length')
    plt.ylabel('GCV Error')
    plt.show()

    plt.clf()
    plt.plot(x, y, '.', label='Data')
    plt.plot(x, ysmooth, label='Smoothed', lw=2)
    plt.legend()
    plt.show()
