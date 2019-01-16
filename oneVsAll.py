import numpy as np

from ann import ANN

class OneVsAll:

    def __init__(self, dp_train, dp_test):
        self.logistic_classifiers = []
        self.dp_train = dp_train
        self.dp_test = dp_test

    def setupClassifiers(self):

        data_dict = self.dp_train.binarize_classes()

        for _, (label, data) in data_dict.items():
            X = data[:, 0:-1]
            y_train_logistic = data[:, -1]
            logistic_classifier = ANN((8,), 500, 0.01, verbose=False)
            logistic_classifier.train(X, y_train_logistic)
            logistic_classifier.score(X, y_train_logistic)
            self.logistic_classifiers.append(logistic_classifier)
            print(label)

