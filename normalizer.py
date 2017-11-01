
from base_builder import BaseBuilder


class Normalizer(BaseBuilder):
    def __init__(self, **kwargs):
        self.cols = kwargs['cols']
        super(Normalizer, self).__init__(**kwargs)

    def _preprocess(self, data):

        for col in self.cols:
            data[:, col] = (data[:, col] - data[:, col].min()) / float(data[:, col].max() - data[:, col].min())

        return data