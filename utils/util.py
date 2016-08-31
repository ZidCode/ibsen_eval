import numpy as np


def construct_weights(std):
    return 1 / (std)


def cut_range(range_, x, y, std):
    start = np.where(x <= range_[0])[0][-1]
    end = np.where(x >= range_[1])[0][0]
    return x[start:end], y[start:end], std[start:end]
