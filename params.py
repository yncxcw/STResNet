'''
Author: Sneha Singhania

This file contains class Params for hyperparameter declarations.
'''

class Params(object):
    batch_size = 32
    map_height = 16
    map_width = 16
    closeness_sequence_length = 3
    period_sequence_length = 3
    trend_sequence_length =  3
    nb_flow = 1
    num_of_filters = 64
    num_of_residual_units = 12
    num_of_output = 1 #depth of predicted output map
    delta = 0.5
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    lr = 0.001
    num_epochs = 50
    logdir = "train"
    data_path = "nova.csv"
