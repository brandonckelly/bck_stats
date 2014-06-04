__author__ = 'brandonkelly'

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time


@jit("f8(f8[:], f8[:])")
def dynamic_time_warping(tseries1, tseries2):
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
            b = dtw[i, j]
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
            dtw[i, j] = (tseries1[i] - tseries2[j]) ** 2 + delta
            path[i, j] = idx

    return dtw[-1, -1], dtw, path


class DBA(object):

    def __init__(self, max_iter, tol=1e-2, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.average = np.zeros(1)
        self.wgss = 0.0  # the within-group sum of squares, called the inertia in the clustering literature
        self.verbose = verbose

    def compute_average(self, tseries, initial_value=None):
        nseries = tseries.shape[0]

        if initial_value is None:
            # initialize the average as a random draw from the set of inputs
            idx = np.random.random_integers(0, nseries-1)
            self.average = tseries[idx]

        # first iteration: get initial within-group sum of squares
        if self.verbose:
            print 'Doing iteration'
            print ' ', '0', '...'
        wgss = self._dba_iteration(tseries)

        # main DBA loop
        for i in range(1, self.max_iter):
            if self.verbose:
                print ' ', i, '...'
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

        return self.average

    def _dba_iteration(self, tseries):
        ntime = tseries.shape[1]
        nseries = tseries.shape[0]

        # table telling us which elements of the time series are identified with a specific element of the DBA average
        assoc_table = []
        for i in range(ntime):
            assoc_table.append([])

        wgss = 0.0  # within group sum of squares from previous iteration, compute here so we don't have to repeat
        for k in range(nseries):
            dtw_dist, dtw, path = dynamic_time_warping(self.average, tseries[k])
            wgss += dtw_dist ** 2
            i = ntime - 1
            j = ntime - 1
            while i >= 0 and j >= 0:
                assoc_table[i].append(tseries[k, j])
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
    nseries = 100
    ntime = 2000
    phase = 0.1 + 0.2 * np.random.uniform(0.0, 1.0, nseries) - 0.1
    period = np.pi / 2.0 + np.pi / 10.0 * np.random.standard_normal(nseries)

    noise_amplitude = 0.0

    t = np.linspace(0.0, 10.0, ntime)
    tseries = np.zeros((nseries, ntime))
    for i in range(nseries):
        tseries[i] = np.sin(t / period[i] + phase[i]) + noise_amplitude * np.random.standard_normal(ntime)

    t1 = time.clock()
    for i in range(1):
        print i
        dtw_dist = dynamic_time_warping(tseries[0], tseries[1])
    t2 = time.clock()
    print 'DTW algorithm tool', t2 - t1, 'seconds.'

    exit()

    niter = 5
    dba = DBA(niter, verbose=True)
    t1 = time.clock()
    dba_avg = dba.compute_average(tseries)
    t2 = time.clock()

    print 'DBA algorithm took', t2 - t1, 'seconds.'

    for i in range(10):
        plt.plot(t, tseries[i])
    plt.plot(t, dba_avg, 'k', lw=4)
    plt.show()