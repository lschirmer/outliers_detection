import numpy as np
from base_builder import BaseBuilder


class Expander(BaseBuilder):
    def __init__(self, num_expansions=1, **kwargs):
        super(Expander, self).__init__(**kwargs)
        self.num_expansions = num_expansions

    def _preprocess(self, data):

        result =()

        for index in range(self.num_expansions):
            result = result + (data,)

        result = np.stack(result, axis=0)

        return result.reshape(data.shape[0], self.num_expansions)


