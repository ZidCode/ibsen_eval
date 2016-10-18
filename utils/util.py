import numpy as np
from scipy.constants import atmosphere


def construct_weights(std):
    return 1 / (std)


def cut_range(range_, x, y, std):
    start = np.where(x <= range_[0])[0][-1]
    end = np.where(x >= range_[1])[0][0]
    return x[start:end], y[start:end], std[start:end]


def international_barometric_formula(height):
    p_0 = atmosphere * 1e-2  # Sea level pressure [hPa]
    a = 0.0065  # Temperature gradient [K/m]
    T = 288.15  # Temperature [K]
    exponent = 5.255
    return p_0 * (1 - (a * height) / T) ** exponent
