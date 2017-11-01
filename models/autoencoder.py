from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Dense
import matplotlib.pyplot as plt
import numpy as np


from base_neural_network import BaseNeuralNetwork


class AutoencoderNetwork(BaseNeuralNetwork):
    def __init__(self, output_dir, **kwargs):
        self.num_features = kwargs['num_features']
        super(AutoencoderNetwork, self).__init__(output_dir=output_dir, **kwargs)

    def _load_model(self):
        inputs = Input((self.num_features,), name='inputs')

        dense1 = Dense(int(self.num_features / 2), activation="sigmoid")(inputs)

        dense2 = Dense(int(self.num_features / 3), activation="sigmoid")(dense1)

        dense3 = Dense(int(self.num_features / 2), activation="sigmoid")(dense2)

        dense4 = Dense(self.num_features, activation="sigmoid")(dense3)

        self.model = Model(inputs=inputs, outputs=dense4)

        self._compile()

    def train(self, training_dataset, validation_dataset):
        return super(AutoencoderNetwork, self).train((training_dataset[0],training_dataset[0]),(validation_dataset[0],validation_dataset[0]))

    def visualize(self, **kwargs):

        X_outliers = kwargs['X_outliers']
        X_validation = kwargs['X_validation']
        y_pred_outliers = kwargs['y_pred_outliers']
        y_pred_validation = kwargs['y_pred_validation']

        euclidian_distance_outliers = np.linalg.norm(X_outliers - y_pred_outliers, axis=1)
        euclidian_distance_validation = np.linalg.norm(X_validation - y_pred_validation, axis=1)

        plt.title('Euclidian distance')
        plt.ylabel('X - y_pred')
        plt.xlabel('Instance number')
        plt.plot(euclidian_distance_outliers, 'ro', euclidian_distance_validation, 'bo')

        plt.show()