import numpy as np
import itertools

from ann import ANN

class OneVsOne:

    def __init__(self, dp_train):
        self.labels_combination = []
        self.logistic_classifiers = []
        self.dp_train = dp_train

    def train(self, hidden_layers, epochs, learning_rate):
        labels = self.dp_train.labels()
        self.labels_combination = list(itertools.combinations(labels, 2))
        for labels in self.labels_combination:
            data = self.dp_train.binarize(labels)
            X = data[:, 0:-1]
            y = data[:, -1]
            logistic_classifier = ANN(hidden_layers, epochs, learning_rate, verbose=False)
            logistic_classifier.train(X, y)
            self.logistic_classifiers.append(logistic_classifier)

    def predict(self, x):
        labels = self.dp_train.labels()
        results = np.zeros((len(labels)))

        for i, classifier in enumerate(self.logistic_classifiers):
            result = int(round(classifier.predict(x)[0]))
            result = self.labels_combination[i][result]
            index = np.where(labels==result)
            results[index] += 1

        index = np.argmax(results)
        return labels[index]



