from ann import ANN
from dataProvider import DataProvider
from oneVsAll import OneVsAll
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import numpy as np

seed = 7
np.random.seed(seed)

def perform_ova_single_nn():
    dp_train = DataProvider('data/rs_train5_tolerances_10_no_headers.txt')
    x_train, y_train = dp_train.split()

    dp_test = DataProvider('data/rs_test5_tolerances_10_no_headers.txt')
    x_test, y_test = dp_test.split()

    ann = ANN((100,), 5000, 0.01, verbose=False)

    parameter_space = {
        'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (50,50), (50,100), (100,50), (200,50)],
        'learning_rate_init': [0.005, 0.01, 0.015, 0.02, 0.025],
        'max_iter': [2000, 3000, 4000]
    }
    grid = GridSearchCV(ann.mlp, parameter_space, n_jobs=-1, cv=3)
    grid.fit(x_train, y_train)
    print('Best parameters found:\n', grid.best_params_)
    print('Results on the test set:\n', classification_report(y_test, grid.predict(x_test)))
    # print('Results on the test set:\n', confusion_matrix(y_test, grid.predict(x_test)))

    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def perform_ova_multiple_nn():
    dp_train = DataProvider('data/rs_train5_tolerances_10_no_headers.txt')
    x_train, y_train = dp_train.split()

    dp_test = DataProvider('data/rs_test5_tolerances_10_no_headers.txt')
    x_test, y_test = dp_test.split()

    ova = OneVsAll(dp_train)
    ova.train((150,), 5000, 0.02)
    y_result = []
    for row in x_test:
        y_result.append(ova.predict(row.reshape(1, -1)))
    print(classification_report(y_test, y_result))
    print(accuracy_score(y_test, y_result))

perform_ova_single_nn()
perform_ova_multiple_nn()