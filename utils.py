'''
Author: Sneha Singhania

This file contains helper functions for running the main neural network.
'''

import numpy as np

def get_dataset_indices(data_len, data_type):
    if data_type == "train":
        start = 0
        end = int(data_len * 0.7)
    else:
        start = int(data_len * 0.7)
        end = data_len
    return np.arange(data_len)[start :end]

def batch_generator(dataloader, batch_size, data_type):
    """
    Batch generator 
    """
    data_len = len(dataloader)
    indices = get_dataset_indices(data_len, data_type)
    start = 0;
    end = len(indices)

    while True:
        if start + batch_size < end:
            closeness_list = []
            period_list = []
            trend_list = []
            predict_list = []
            for j in range(batch_size):
                index = indices[start + j]
                closeness, period, trend, predict = dataloader[index]
                closeness_list.append(closeness)
                period_list.append(period)
                trend_list.append(trend)
                predict_list.append(predict)
            yield np.stack(closeness_list), np.stack(period_list), np.stack(trend_list), np.stack(predict_list)
            start += batch_size 
        else:
            start = 0
            end = len(indices)
            continue
