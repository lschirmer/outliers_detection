import sklearn.preprocessing as preprocessing

from base_builder import BaseBuilder


class OneHotEncoder(BaseBuilder):
    def __init__(self, **kwargs):
        self.cols = kwargs['cols']
        super(OneHotEncoder, self).__init__(**kwargs)

    def _preprocess(self, data):
        encoder = preprocessing.OneHotEncoder(categorical_features=self.cols)
        result = encoder.fit_transform(data).toarray()
        return result
