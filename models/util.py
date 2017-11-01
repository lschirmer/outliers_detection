import keras.backend as K
import os


def precision(y_true, y_pred):
    y_true_rounded = K.clip(y_true, 0, 1)
    y_pred_rounded = K.round(K.clip(y_pred, 0, 1))
    true_positives = K.sum(y_true_rounded * y_pred_rounded)
    predicted_positives = K.sum(y_pred_rounded)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    y_true_rounded = K.clip(y_true, 0, 1)
    y_pred_rounded = K.round(K.clip(y_pred, 0, 1))
    true_positives = K.sum(y_true_rounded * y_pred_rounded)
    possible_positives = K.sum(y_true_rounded)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    precision_measure = precision(y_true, y_pred)
    recall_measure = recall(y_true, y_pred)

    return 2 * ((precision_measure * recall_measure) / (precision_measure + recall_measure + K.epsilon()))


def create_dirs(output_dir):
    if not os.path.exists(output_dir + "/models"):
        os.makedirs(output_dir + "/models")