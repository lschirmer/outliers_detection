import sklearn.metrics
import argparse
import numpy as np
import csv
import scipy
from sklearn import manifold
from loader import *
from preprocess import *
from sksos import SOS

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn
import pickle
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor

data_loader = {
    'csv' : CSVLoader,
}

def pcaProj(data):
    return 0

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def m_gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area


def tsneProj(data, outlier_indexes,index):

    print("outliers",outlier_indexes)

    r = np.random.uniform(0,data.shape[0],3000).astype(int)

    m_data = []
    m_outliers = []
    m_index = []
    for i in r:
        m_data.append(data[i,:])
        m_index.append((index[i]))
#        m_outliers.append(outlier_indexes[i,0])

    out_data = np.copy(m_data)

    detector = SOS()
    y_pred = detector.predict(out_data)

    print("pred",out_data)
    g_score = gini(y_pred)
    print("m_gini",g_score)


    with open("out.csv", "w") as file:
        writer = csv.writer(file, delimiter=",", lineterminator="\n")
        writer.writerow(['id', 'target'])

        for i in range(len(m_index)):
            row = [m_index[i]]
            row.extend(np.ravel(y_pred[i]).tolist())
            writer.writerow(row)

    return m_data


if __name__ == "__main__":

    loader_class = data_loader['csv']
    loader = loader_class(path= './test.csv' , y_col_name='target', remove_cols=[], primary_key_col = 'id')
    X, y, index = loader.load_data()

    # to calculate precision after
    outlier_indexes = np.where(y == 1)

    print(outlier_indexes)

    normalizer_column_indexes = [loader.data.columns.get_loc(c) for c in loader.data.columns if not (c.endswith('cat') or c.endswith('bin'))]
    one_hot_encoder_column_indexes = [loader.data.columns.get_loc(c) for c in loader.data.columns if (c.endswith('cat'))]
    X_preprocessed = Normalizer(
        cols=normalizer_column_indexes,
        next= Shifter(
            next=OneHotEncoder(
                cols=one_hot_encoder_column_indexes
            )
        )
    ).preprocess(X)

    num_features = X_preprocessed.shape[1]
    print(X_preprocessed.shape[0],X_preprocessed)

    data = tsneProj(X_preprocessed, y,index)


