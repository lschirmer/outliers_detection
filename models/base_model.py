
class BaseModel(object):
    def __init__(self, **kwargs):
        self.params = kwargs


    def load_model(self, model_dir):
        raise NotImplementedError('Not implemented')


    def train(self, training_dataset, validation_dataset):
        raise NotImplementedError('Not implemented')


    def predict(self, X):
        raise NotImplementedError('Not implemented')


    def visualize(self, **kwargs):
        raise NotImplementedError('Not implemented')