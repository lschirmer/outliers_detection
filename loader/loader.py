

class Loader(object):
    def __init__(self, path):
        self.path = path

    def load_data(self):
        return self._load()

    def _load(self):
        raise NotImplementedError("Not implemented")