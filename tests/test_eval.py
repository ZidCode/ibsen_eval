import os
import numpy as np
from datetime import datetime
from numpy.testing import assert_array_equal
import evaluation.evaluation as ie
from test_config_dict import test_config


def test_parse_ini_config():
    GPS = [53.9453236, 11.3829424, 0]
    FITTING_VALS = ['range_', 'params', 'initial_values', 'limits']
    filename = 'test_config.ini'
    config = ie.parse_ini_config(filename)
    assert_array_equal(config['Processing']['gps_coords'], GPS)
    assert_array_equal(sorted(config['Fitting'].keys()), sorted(FITTING_VALS))

def test_evaluate_spectra():
    ie.evaluate_spectra(test_config)

