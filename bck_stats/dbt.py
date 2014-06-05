__author__ = 'brandonkelly'

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time


@jit
def dynamic_time_warping(tseries1, tseries2):
    """
    Compute the dynamic time warping (DTW) distance between two time series. It is assumed that the time series are
    evenly sampled, but they can have different lengths. Numba is used to speed up the computation, so you must have
    Numba installed.

    :param tseries1: The first time series, a numpy array.
    :param tseries2: The second time series, a numpy array.
    :return: A tuple containing the DTW distance, the DTW matrix, and the path matrix taken by the algorithm.
    """
    dtw = np.zeros((len(tseries1), len(tseries2)), dtype=np.float)  # matrix of coordinate distances
    path = np.zeros((len(tseries1), len(tseries2)), dtype=np.int)  # path of algorithm

    # initialize the first row and column
    dtw[0, 0] = (tseries1[0] - tseries2[0]) ** 2
    path[0, 0] = -1

    for i in range(1, len(tseries1)):
        dtw[i, 0] = dtw[i-1, 0] + (tseries1[i] - tseries2[0]) ** 2
        path[i, 0] = 2

    for j in range(1, len(tseries2)):
        dtw[0, j] = dtw[0, j-1] + (tseries1[0] - tseries2[j]) ** 2
        path[0, j] = 1

    # main loop of the DTW algorithm
    for i in range(1, len(tseries1)):
        for j in range(1, len(tseries2)):
            a = dtw[i-1, j-1]
            b = dtw[i, j-1]
            c = dtw[i-1, j]
            if a < b:
                if a < c:
                    idx = 0  # a is the minimum
                    delta = a
                else:
                    idx = 2  # c is the minimum
                    delta = c
            else:
                if b < c:
                    idx = 1  # b is the minimum
                    delta = b
                else:
                    idx = 2  # c is the minimum
                    delta = c
            # neighbors = np.array([dtw[i-1, j-1], dtw[i, j-1], dtw[i-1, j]])
            # idx = np.argmin(neighbors)
            # delta = neighbors[idx]
            dtw[i, j] = (tseries1[i] - tseries2[j]) ** 2 + delta
            path[i, j] = idx

    return dtw[-1, -1], dtw, path


class DBA(object):

    def __init__(self, max_iter, tol=1e-4, verbose=False):
        """
        Constructor for the DBA class. This class computes the dynamic time warping (DTW) barycenter averaging (DBA)
        strategy for averaging a set of evenly-sampled time series. The method is described in

        "A global averaging method for dynamic time warping, with applications to clustering." Petitjean, F.,
            Ketterlin, A., & Gancarski, P. 2011, Pattern Recognition, 44, 678-693.

        :param max_iter: The maximum number of iterations for the DBA algorithm.
        :param tol: The tolerance level for the algorithm. The algorithm terminates once the fractional difference in
            the within-group sum of squares between successive iterations is less than tol. The algorithm will also
            terminate if the maximum number of iterations is exceeded, or if the sum of squares increases.
        :param verbose: If true, then provide helpful output.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.average = np.zeros(1)
        self.wgss = 0.0  # the within-group sum of squares, called the inertia in the clustering literature
        self.verbose = verbose

    def compute_average(self, tseries, nstarts=1, initial_value=None, dba_length=None):
        """
        Perform the DBA algorithm to compute the average for a set of time series. The algorithm is a local optimization
        strategy and thus depends on the initial guess for the average. Improved results can be obtained by using
        multiple random initial starts.

        :param tseries: The list of time series, a list of numpy arrays.
        :param nstarts: The number of random starts to use for the DBA algorithm. The average time series that minimizes
            the within-group sum of squares over the random starts is returned and saved.
        :param initial_value: The initial value for the DBA algorithm, a numpy array. If None, then the initial values
             will be drawn randomly from the set of input time series (recommended). Note that is an initial guess is
             supplied, then the nstarts parameter is ignored.
        :param dba_length: The length of the DBA average time series. If None, this will be set to the length of the
            initial_value array. Otherwise, the initial value array will be linearly interpolated to this length.
        :return: The estimated average of the time series, defined to minimize the within-group sum of squares of the
            input set of time series.
        """
        if initial_value is not None:
            nstarts = 1

        if initial_value is None:
            # initialize the average as a random draw from the set of inputs
            start_idx = np.random.permutation(len(tseries))[:nstarts]

        best_wgss = 1e300
        if self.verbose:
            print 'Doing initialization iteration:'
        for i in range(nstarts):
            print i, '...'
            if initial_value is None:
                iseries = tseries[start_idx[i]]
            else:
                iseries = initial_value
            if dba_length is not None:
                # linearly interpolate initial average value to the requested length
                lininterp = interp1d(np.arange(len(iseries)), iseries)
                iseries = lininterp(np.arange(dba_length))

            self._run_dba(tseries, iseries)

            if self.wgss < best_wgss:
                # found better average, save it
                if self.verbose:
                    print 'New best estimate found for random start', i
                best_wgss = self.wgss
                best_average = self.average

        self.wgss = best_wgss
        self.average = best_average

        return best_average

    def _run_dba(self, tseries, initial_value):
        """ Perform the DBA algorithm. """
        nseries = len(tseries)

        self.average = initial_value

        # first iteration: get initial within-group sum of squares
        if self.verbose:
            print 'Doing iteration'
            print ' ', '0', '...'
        wgss = self._dba_iteration(tseries)

        # main DBA loop
        for i in range(1, self.max_iter):
            if self.verbose:
                print ' ', i, '...', 'WGSS:', wgss
            wgss_old = wgss
            # WGSS is actually from previous iteration, but don't compute again because it is expensive
            wgss = self._dba_iteration(tseries)
            if wgss > wgss_old:
                # sum of squares should be non-increasing
                print 'Warning! Within-group sum of squares increased at iteration', i, 'terminating algorithm.'
                break
            elif np.abs(wgss - wgss_old) / wgss_old < self.tol:
                # convergence
                break

        # compute final within-group sum of squares
        wgss = 0.0
        for k in range(nseries):
            wgss += dynamic_time_warping(tseries[k], self.average)[0] ** 2
        self.wgss = wgss

    def _dba_iteration(self, tseries):
        """ Perform a single iteration of the DBA algorithm. """
        ntime = len(self.average)

        # table telling us which elements of the time series are identified with a specific element of the DBA average
        assoc_table = []
        for i in range(ntime):
            assoc_table.append([])

        wgss = 0.0  # within group sum of squares from previous iteration, compute here so we don't have to repeat
        for series in tseries:
            dtw_dist, dtw, path = dynamic_time_warping(self.average, series)
            wgss += dtw_dist ** 2
            i = ntime - 1
            j = len(series) - 1
            while i >= 0 and j >= 0:
                assoc_table[i].append(series[j])
                if path[i, j] == 0:
                    i -= 1
                    j -= 1
                elif path[i, j] == 1:
                    j -= 1
                elif path[i, j] == 2:
                    i -= 1
                else:
                    # should not happen, but just in case make sure we bail once path[i, j] = -1
                    break

        # update the average
        for i, cell in enumerate(assoc_table):
            self.average[i] = np.mean(cell)

        return wgss


if __name__ == "__main__":
    # run on some test data
    nseries = 40
    ntime0 = 1000
    phase = 0.1 + 0.2 * np.random.uniform(0.0, 1.0, nseries) - 0.1
    period = np.pi / 4.0 + np.pi / 100.0 * np.random.standard_normal(nseries)

    noise_amplitude = 0.0

    t_list = []
    ts_list = []
    for i in range(nseries):
        ntime = np.random.random_integers(ntime0 * 0.9, ntime0 * 1.1)
        t = np.linspace(0.0, 10.0, ntime)
        t_list.append(t)
        tseries = np.sin(t / period[i] + phase[i]) + noise_amplitude * np.random.standard_normal(ntime)
        ts_list.append(tseries)

    niter = 30
    dba = DBA(niter, verbose=True, tol=1e-4)
    t1 = time.clock()
    dba_avg = dba.compute_average(ts_list, nstarts=5)
    t2 = time.clock()

    print 'DBA algorithm took', t2 - t1, 'seconds.'

    for i in range(nseries):
        plt.plot(t_list[i], ts_list[i], '.', ms=2)
    t = np.linspace(0.0, 10.0, len(dba_avg))
    plt.plot(t, dba_avg, 'k', lw=3)
    plt.show()