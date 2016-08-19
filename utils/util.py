import numpy as np


def construct_weights(x, range_):

    weights = np.ones(len(x))
    index_ = np.concatenate((np.where(x < min(range_))[0], np.where(x > max(range_))[0]))
    weights[index_] = 0
    return weights
