__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from sklearn_estimator_suite import ClassificationSuite
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

n_samples = 2000
n_classes = 3
X, y = make_classification(n_samples, n_classes=n_classes, n_informative=10)

X, X_test, y, y_test = train_test_split(X, y, train_size=0.5)

# suite = ClassificationSuite(n_features=X.shape[1])
#
# suite.fit(X, y)
# names = suite.best_scores.keys()
# scores = suite.best_scores.values()
#
# fig, ax1 = plt.subplots()
# plt.bar(np.arange(0, len(names)), scores)
# xtickNames = plt.setp(ax1, xticklabels=names)
# plt.setp(xtickNames, rotation=45)
# plt.ylabel('Accuracy')
# plt.xlabel('Model')
# plt.show()

# now make sure things work in parallel
suite = ClassificationSuite(n_features=X.shape[1], njobs=7)

suite.fit(X, y)

names = suite.best_scores.keys()
scores = suite.best_scores.values()

# get predictions
y_predict_uniform = suite.predict(X_test, weights='uniform')  # uniform weightings
y_predict_stacked = suite.predict(X_test)

uniform_accuracy = accuracy_score(y_test, y_predict_uniform)
stacked_accuracy = accuracy_score(y_test, y_predict_stacked)
y_predict_all = suite.predict_all(X_test)

print ''
print '---'
print 'Test accuracy for uniform weighting:', uniform_accuracy
print 'Test accuracy for validation score weighting:', stacked_accuracy
for name in y_predict_all:
    print 'Test accuracy for', name, ':', accuracy_score(y_test, y_predict_all[name])
print '---'
print ''

fig, ax1 = plt.subplots()
plt.bar(np.arange(0, len(names)), scores)
xtickNames = plt.setp(ax1, xticklabels=names)
plt.setp(xtickNames, rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.show()

# try using different number of grid refinements for the models
n_refinements = {name: 1 for name in suite.model_names}
n_refinements['GbcAutoNtrees'] = 0

suite.fit(X, y, n_refinements=n_refinements)

names = suite.best_scores.keys()
scores = suite.best_scores.values()

fig, ax1 = plt.subplots()
plt.bar(np.arange(0, len(names)), scores)
xtickNames = plt.setp(ax1, xticklabels=names)
plt.setp(xtickNames, rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.tight_layout()
plt.show()

tuning_ranges = {'LogisticRegression': {'C': list(np.logspace(-3.0, 0.0, 5))},
                 'DecisionTreeClassifier': {'max_depth': [5, 10, 20, 50, 100]},
                 'LinearSVC': {'C': list(np.logspace(-3.0, 0.0, 5))}}

suite = ClassificationSuite(tuning_ranges=tuning_ranges, njobs=7)

suite.fit(X, y, n_refinements=3)

names = suite.best_scores.keys()
scores = suite.best_scores.values()

fig, ax1 = plt.subplots()
plt.bar(np.arange(0, len(names)), scores)
xtickNames = plt.setp(ax1, xticklabels=names)
plt.setp(xtickNames, rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.show()

