
import matplotlib.pyplot as plt
import seaborn as sns
import snips as snp

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class ANN:

    def __init__(self, hidden_layers, epochs, learning_rate, verbose=True):
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation="logistic", solver="sgd", max_iter=epochs, learning_rate_init=learning_rate, verbose=verbose)

    def train(self, x, y):
        self.mlp.fit(x, y)

    def predict(self, x):
        return self.mlp.predict(x)

    def score(self, x, y):
        y_pred = self.predict(x)
        print(accuracy_score(y, y_pred))
        # cm = confusion_matrix(y, y_pred)
        # print(cm)
        # sns.heatmap(cm, center=True)
        # plt.show()
