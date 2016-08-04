#!/usr/bin/env python
import numpy as np
from math import cos, radians
'''
This module determines the atmospheric path length or Air mass
due to Young (1994)
'''


def get_atmospheric_path_length(zenith):

    a = 1.002432
    b = 0.148386
    c = 0.0096467
    aa = 0.149864
    bb = 0.0102963
    cc = 0.000303978
    theta_rad = np.radians(zenith)
    AM = (a * cos(theta_rad)**2 + b * cos(theta_rad) + c) / \
         (cos(theta_rad) ** 3 + aa * cos(theta_rad)**2 + bb * cos(theta_rad) + cc)
    return AM
