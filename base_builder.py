


class BaseBuilder(object):
    def __init__(self, next= None, **kwargs):
        self.next = next

    def preprocess(self, data):
        _data = self._preprocess(data)
        if self.next:
            return self.next.preprocess(_data)
        else:
            return _data

    def _preprocess(self, data):
        raise NotImplementedError('Not implemented')