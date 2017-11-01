import pandas as pd
import numpy as np
import h5py

from loader import Loader


class MATLoader(Loader):
    def __init__(self, path, **kwargs):
        super(MATLoader, self).__init__(path=path)

    def _load(self):
        arrays = {}
        f = h5py.File(self.path)
        for k, v in f.items():
            arrays[k] = np.array(v)


        X = arrays['X'].reshape(arrays['X'].shape[1], arrays['X'].shape[0])
        y = arrays['y'].reshape(arrays['y'].shape[1], arrays['y'].shape[0])

        return X, y