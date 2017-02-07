import numpy as np
from scipy.constants import atmosphere
from tempfile import mkstemp
from datetime import datetime



convert2datetime = lambda d: datetime.strptime(d, '%Y-%m-%d %H:%M:%S')

def create_meas_file(DEFAULT_MEAS):
    file_ = mkstemp()

    filename = file_[1]
    with open(filename, 'w') as fp:
        fp.write(DEFAULT_MEAS)
    return filename


def international_barometric_formula(height):
    p_0 = atmosphere * 1e-2  # Sea level pressure [hPa]
    a = 0.0065  # Temperature gradient [K/m]
    T = 288.15  # Temperature [K]
    exponent = 5.255
    return p_0 * (1 - (a * height) / T) ** exponent
