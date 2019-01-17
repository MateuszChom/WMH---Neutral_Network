import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)

from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

class DataProvider:

    def __init__(self, path):
        self.df = pd.read_fwf(path)
        self.df = self.df.values

    def labels(self):
        return np.unique(self.df[:, -1])

    def split(self, normalize=True):
        x = self.df[:, 0:-1]
        if normalize:
            scaler = MinMaxScaler(feature_range=(0, 10))
            x = scaler.fit_transform(x)
        y = self.df[:,-1]

        return x, y

    def binarize(self, label, upsampled=False):
        x, y = self.split()
        data = np.hstack((x, y.reshape(-1,1)))
        data_p = data[data[:, -1] == label]
        data_p[:, -1] = 1
        data_n = data[data[:, -1] != label]
        data_n[:, -1] = 0

        if upsampled:
            data_p = resample(data_p, replace=True, n_samples=data_n.shape[0])
            data_p[:, -1] = 1
        data = np.vstack((data_p, data_n))

        return data

