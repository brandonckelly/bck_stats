bck_stats
=========

Routines for implementing various statistical and machine learning techniques.

Description of routines:

* `super_pca`: Class for performing supervised principal components regression (Bair, E., et al. *Prediction by supervised principal components.* J. Am. Stat. Assoc. 101, 473, 2006)
* `sklearn_estimator_suite`: Classes for running through a set of scikit-learn estimators, using cross-validation to choose the tuning parameters.
* `react`: Classes for performing non-parameteric regression in one or two dimensions based on the REACT technique (Beran, R. *REACT scatterplot smoothers: Superefficiency through basis economy.* J. Am. Stat. Assoc. 95, 449, 2000)
* `multiclass_triangle_plot`: Plot the lower triangle of a scatterplot matrix, color-coding according to class label. A modified version of Dan Foreman-Mackey's triangle.py routine.
* `gcv_smoother`: Perform exponential smoothing of a time series. The e-folding time scale is chosen using generalized cross-validation.
* `dynamic_linear_model`: Class to perform dynamic linear regression via least-squares (Montana, G., et al. *Flexible least squares for temporal data mining and statistical arbitrage.* Expert Systems with Applications 36, 2819, 2009).

-------------
Installation
-------------

From the base directory type `python setup.py install` in a terminal.
