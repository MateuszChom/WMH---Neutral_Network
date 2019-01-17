from ann import ANN
from dataProvider import DataProvider
from oneVsAll import OneVsAll

import numpy as np

dp_train = DataProvider('data/rs_train5_exhaustive_no_headers.txt')
dp_test = DataProvider('data/rs_test5_exhaustive_no_headers.txt')
x_test, y_test = dp_test.split()

ova = OneVsAll(dp_train, dp_test)
ova.setupClassifiers()
ova.predict(x_test[40,:].reshape(1, -1), y_test[40])
# x_test, y_test = dp_test.oneVsAll()
