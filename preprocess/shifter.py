import sklearn.preprocessing as preprocessing

from base_builder import BaseBuilder


class Shifter(BaseBuilder):
    def __init__(self, **kwargs):
        super(Shifter, self).__init__(**kwargs)

    def _preprocess(self, data):
        for index in range(data.shape[1]):
            if data[:,index].min() != 0.:
                data[:,index] = data[:,index] - data[:,index].min()

        assert (data.min() == 0)

        return data

