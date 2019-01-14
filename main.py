from ann import ANN
from dataProvider import DataProvider

import numpy as np

dp_train = DataProvider('data/rs_train5_exhaustive_no_headers.txt')
x_train, y_train = dp_train.oneVsAll()
print(x_train, y_train)

dp_test = DataProvider('data/rs_test5_exhaustive_no_headers.txt')
x_test, y_test = dp_test.oneVsAll()

ann = ANN((150,100), 2000, 0.01)
ann.train(x_train, y_train)
ann.score(x_test, y_test)