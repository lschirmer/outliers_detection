import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import SGD
import keras.backend as K
from keras.metrics import binary_accuracy
import util
from base_model import BaseModel


K.set_image_dim_ordering('th')

metrics =  [binary_accuracy, util.precision, util.recall, util.f1]


class BaseNeuralNetwork(BaseModel):
    def __init__(self, output_dir, **kwargs):
        self.output_dir = output_dir
        self.params = kwargs
        self._load_model()
        super(BaseNeuralNetwork, self).__init__(output_dir=output_dir, **kwargs)

    def _create_model_structure_and_get_callback(self):
        util.create_dirs(self.output_dir)

        models_dir = self.output_dir + "/models/"

        # Saves model architecture
        json_string = self.model.to_json()
        open(models_dir + '/model_architecture.json', 'w').write(json_string)

        checkpoint = ModelCheckpoint(filepath=models_dir + '/model_weights_l{val_loss:.5f}.h5', monitor='val_loss',
                                     mode='min', verbose=1, save_best_only=True)

        return checkpoint

    def _compile(self):
        optimizer = SGD(lr= self.params['learning_rate'], decay=self.params['decay'], momentum=self.params['momentum'], nesterov=self.params['nesterov'])

        self.model.compile(optimizer=optimizer, loss=self.params['loss'], metrics=metrics)

    def load_model(self, model_dir):
        # Loads model architecture
        model_arch_file = open(model_dir + '/model_architecture.json', 'r')
        model_arch_json = model_arch_file.read()
        model_arch_file.close()
        self.model = model_from_json(model_arch_json)

        self._compile()

        # Loads model weights
        self.model.load_weights(model_dir + '/model_weights.h5')

    def train(self, training_dataset, validation_dataset):
        callbacks = []

        model_callback = self._create_model_structure_and_get_callback()

        callbacks.append(model_callback)

        self.model.fit(training_dataset[0], training_dataset[1],
                  batch_size= self.params['batch_size'], epochs=self.params['epochs'], verbose=2,
                  validation_data=validation_dataset, callbacks=callbacks)

    def predict(self, X):
        evaluation_batch_size = 1

        y_pred_list = []
        for x in X:
            x = x.reshape(1, x.shape[0])

            y_pred = self.model.predict(x=x, batch_size=evaluation_batch_size, verbose=0)

            y_pred_list.append(y_pred[0])

        return np.array(y_pred_list)

    def visualize(self, **kwargs):
        raise NotImplementedError('Not implemented')