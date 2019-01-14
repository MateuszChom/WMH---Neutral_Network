import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

class DataProvider:

    def __init__(self, path):
        self.df = pd.read_fwf(path)
        self.df = self.df.values

    def split(self):
        scaler = MinMaxScaler(feature_range=(0, 20))
        x = self.df[:, 0:-1]
        x = scaler.fit_transform(x)
        y = self.df[:,-1]
        return x, y

    def oneVsAll(self):
        x, y = self.split()
        classes = np.unique(y)
        array = np.zeros((len(y), len(classes)))
        for (row, val) in enumerate(y):
            column = np.argwhere(classes == val).flatten()[0]
            array[row][column] = 1

        return x, array

