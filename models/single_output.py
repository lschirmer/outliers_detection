from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Dense
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import plotly.plotly as py
import plotly.graph_objs as go

from base_neural_network import BaseNeuralNetwork


class SingleOutputNetwork(BaseNeuralNetwork):
    def __init__(self, output_dir, **kwargs):
        self.outputs = kwargs['outputs']
        self.num_features = kwargs['num_features']
        super(SingleOutputNetwork, self).__init__(output_dir=output_dir, **kwargs)

    def _load_model(self):
        inputs = Input((self.num_features,), name='inputs')

        dense1 = Dense(6, activation="sigmoid")(inputs)

        dense2 = Dense(self.outputs, activation="sigmoid")(dense1)

        self.model = Model(inputs=inputs, outputs=dense2)

        self._compile()

    def visualize(self, **kwargs):
        y_pred_validation = [kwargs['y_pred_validation'][:,index] for index in range(self.outputs)]
        y_pred_outliers = [kwargs['y_pred_outliers'][:,index] for index in range(self.outputs)]

        if self.outputs == 1:
            plt.title('Sigmoid single output')
            plt.ylabel('y_pred')
            plt.xlabel('Instance number')
        elif self.outputs == 2:
            plt.title('Sigmoid single output')
            plt.xlabel('y_pred[0]')
            plt.ylabel('y_pred[1]')
        elif self.outputs != 3:
            plt.title('Sigmoid single output')
            plt.xlabel('Instance number')
            plt.ylabel('Euclidean distance')
            y_pred_outliers = [np.linalg.norm(kwargs['y_pred_outliers'], axis=1)]
            y_pred_validation = [np.linalg.norm(kwargs['y_pred_validation'], axis=1)]

        if self.outputs == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', title='y_pred')
            ax.scatter(c="r", marker="o", *y_pred_outliers)
            ax.scatter(c="b", marker="o", *y_pred_validation)
        else:
            plt.plot(color='b', linestyle=' ', marker='o', *y_pred_validation)
            plt.plot(color='r', linestyle=' ', marker='o', *y_pred_outliers)


        plt.show()



        # Create a trace
        # trace_validation = go.Scatter(
        #     x=y_pred_validation,
        #     mode='markers',
        #     marker = {
        #         'color': 'rgb(0,0,255)'
        #     }
        # )
        #
        # trace_outliers = go.Scatter(
        #     x=y_pred_outliers,
        #     mode='markers',
        #     marker={
        #         'color': 'rgb(255,0,0)'
        #     }
        # )

        # data = [trace_validation, trace_outliers]

        # Plot and embed in ipython notebook!
        # py.plot(data)


