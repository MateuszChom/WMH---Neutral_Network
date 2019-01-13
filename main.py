from ann import ANN
from dataProvider import DataProvider

dp_train = DataProvider('data/rs_train5_exhaustive_no_headers.txt')
x_train, y_train = dp_train.split()

dp_test = DataProvider('data/rs_test5_exhaustive_no_headers.txt')
x_test, y_test = dp_test.split()

ann = ANN((150,100), 3000, 0.01)
ann.train(x_train, y_train)
ann.score(x_test, y_test)
