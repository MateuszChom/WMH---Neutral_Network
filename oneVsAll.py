import numpy as np

from ann import ANN

class OneVsAll:

    def __init__(self, dp_train):
        self.logistic_classifiers = []
        self.dp_train = dp_train

    def train(self, hidden_layers, epochs, learning_rate):
        labels = self.dp_train.labels()
        for label in labels:
            data = self.dp_train.binarize(label, upsampled=True)
            X = data[:, 0:-1]
            y_train_logistic = data[:, -1]
            logistic_classifier = ANN(hidden_layers, epochs, learning_rate, verbose=False)
            logistic_classifier.train(X, y_train_logistic)
            # logistic_classifier.score(X, y_train_logistic)
            self.logistic_classifiers.append(logistic_classifier)
            
    def predict(self, x):
        results = []
        labels = self.dp_train.labels()
        for classifier in self.logistic_classifiers:
            results.append(classifier.predict(x))
        index = np.argmax(results)
        
        return labels[index]


