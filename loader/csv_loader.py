import pandas as pd
import numpy as np


class Loader(object):
    def __init__(self, path):
        self.path = path

    def load_data(self):
        return self._load()

    def _load(self):
        raise NotImplementedError("Not implemented")


class CSVLoader(Loader):
    def __init__(self, path, y_col_name, remove_cols, primary_key_col='id', sep=',', header=0, ):
        self.sep = sep
        self.header = header
        self.y_col_name = y_col_name
        self.primary_key_col = primary_key_col
        self.remove_cols = remove_cols
        self.data = None
        super(CSVLoader, self).__init__(path=path)

    def _load(self):
        self.data = pd.read_csv(self.path, sep=self.sep, header=self.header)

        np_index = np.array(self.data[self.primary_key_col])

        np_y = np.array(self.data[self.y_col_name])
        np_y = np_y.reshape(np_y.shape[0], 1)

        del self.data[self.y_col_name]
        del self.data[self.primary_key_col]
        for col in self.remove_cols:
            del self.data[col]

        np_X = np.array(self.data)

        return np_X, np_y, np_index
