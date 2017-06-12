# coding: utf-8

'''
cross-validation 데이터 셋을 만들때 y 값 분포가 동일한 비율로 유지되면서 샘플링 되도록 한다.
'''

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

X = np.array([['a'], ['b'], ['c'], ['d'], ['e'], ['f'], ['g'], ['h'], ['j'], ['k'], ['l'], ['m'], ['n'], ['o'], ['p'], ['q'], ['r'], ['s'], ['t'], ['u']])
y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

s = StratifiedShuffleSplit(n_splits=10, test_size=0.5, train_size=None, random_state=None)
print "# of cross-validation dataset : ", s.get_n_splits(X, y) # Returns the number of splitting iterations in the cross-validator.

for train_index, test_index in s.split(X, y): # Generate indices to split data into training and test set.
	print "TRAIN:", train_index, "TEST:", test_index
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

