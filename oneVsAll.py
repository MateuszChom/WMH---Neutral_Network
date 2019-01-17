import numpy as np

from ann import ANN

class OneVsAll:

    def __init__(self, dp_train, dp_test):
        self.logistic_classifiers = []
        self.dp_train = dp_train
        self.dp_test = dp_test

    def setupClassifiers(self):
        labels = self.dp_train.labels()
        for label in labels:
            data = self.dp_train.binarize(label, upsampled=True)
            X = data[:, 0:-1]
            y_train_logistic = data[:, -1]
            logistic_classifier = ANN((10,), 400, 0.01, verbose=False)
            logistic_classifier.train(X, y_train_logistic)
            logistic_classifier.score(X, y_train_logistic)
            self.logistic_classifiers.append(logistic_classifier)
            
    def predict(self, x, y):
        results = []
        labels = self.dp_train.labels()
        for classifier in self.logistic_classifiers:
            results.append(classifier.predict(x))
        print(labels, results)
        index = np.argmax(results)
        print(labels[index], y)


