__author__ = 'brandonkelly'
__notes__ = "Adapted from Dan Foreman-Mackey triangle.py module."

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def multiclass_triangle(xs, classes, labels=None, verbose=True, fig=None, **kwargs):
    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"

    K = len(xs)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.05 * factor  # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig.axes), K))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    extents = [[x.min(), x.max()] for x in xs]

    # Check for parameters that never change.
    m = np.array([e[0] == e[1] for e in extents], dtype=bool)
    if np.any(m):
        raise ValueError(("It looks like the parameter(s) in column(s) "
                          "{0} have no dynamic range. Please provide an "
                          "`extent` argument.")
                         .format(", ".join(map("{0}".format,
                                               np.arange(len(m))[m]))))

    class_labels = np.unique(classes)
    nclasses = len(class_labels)

    color_list = ["Black", "DodgerBlue", "DarkOrange", "Green", "Magenta", "Red", "Brown", "Cyan"] * 10

    for i, x in enumerate(xs):
        ax = axes[i, i]
        # Plot the histograms.
        n = []
        for l, k in enumerate(class_labels):
            n_k, b_k, p_k = ax.hist(x[classes == k], bins=kwargs.get("bins", 50),
                                    range=extents[i], histtype="step",
                                    color=color_list[l], lw=2, normed=True)
            n.append(n_k)

        # Set up the axes.
        ax.set_xlim(extents[i])
        ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(5))

        # Not so DRY.
        if i < K - 1:
            ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
                ax.xaxis.set_label_coords(0.5, -0.3)

        for j, y in enumerate(xs):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                continue

            for l, k in enumerate(class_labels):
                ax.plot(y[classes == k], x[classes == k], 'o', ms=1.5, color=color_list[l], rasterized=True, alpha=0.25)

            extent = [[y.min(), y.max()], [x.min(), x.max()]]
            ax.set_xlim(extent[0])
            ax.set_ylim(extent[1])
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
                    ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.3, 0.5)

    return fig