import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer

class DataProvider:

    def __init__(self, path):
        self.df = pd.read_fwf(path)
        self.df = self.df.values

    def classes(self):
        return np.unique(self.df[:, -1])

    def split(self, normalize=True):
        x = self.df[:, 0:-1]
        if normalize:
            scaler = MinMaxScaler(feature_range=(0, 1)) #StandardScaler() #MinMaxScaler(feature_range=(-1, 1))
            x = scaler.fit_transform(x)
        y = self.df[:,-1]

        return x, y

    def binarize_classes(self, balanced=True):
        x, y = self.split()
        data = np.hstack((x,y.reshape(-1,1)))
        dict = {}
        for i, label in enumerate(self.classes()):
            data_p = data[data[:, -1] == label]
            data_p[:, -1] = 1
            data_n = data[data[:, -1] != label]
            data_n = data_n[0:data_p.shape[0]]
            data_n[:, -1] = 0

            dict[i] = (label, np.vstack((data_p, data_n)))

        return dict

