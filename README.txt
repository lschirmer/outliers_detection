########################
#                      #
#     Requirements     #
#                      #
########################

python 2.7
numpy
pandas
keras
h5py
scikit-learn


#################################
#                               #
#     Command line examples     #
#                               #
#################################

# Trains an autoencoder
> python run.py -d csv -m autoencoder -o out/autoencoder/output_001 train data/train.csv

# Visualizes the classification of a single_nn model
> python run.py -d csv -m single_nn -l out/single_nn/1/output_001/models -s 1 visualize data/train.csv

# Evaluates the classification of a single_nn model
> python run.py -d csv -m single_nn -l out/single_nn/1/output_001/models -s 1 -t 0.3 evaluate data/train.csv

# Generates a prediction file with outputs of a single_nn model
> python run.py -d csv -m single_nn -l out/single_nn/1/output_001/models -f out/single_nn/1/output_001/prediction.csv -s 1 predict data/train.csv

