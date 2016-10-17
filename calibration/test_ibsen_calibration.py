import os
import copy
import numpy as np
from  numpy.testing import assert_array_equal
from ibsen_calibration import sort_ibsen_by_int, subtract_dark
from extract_nonlinearity import calculate_nonlinearity_factors, generate_nonlinear_correction
from extract_response import generate_response_factors

IntTime = np.array([260.0, 5.0, 10.0, 140.0, 15.0, 20.0, 280.0, 25.0, 30.0,
                    160.0, 35.0, 40.0, 45.0, 50.0, 180.0, 60.0, 70.0, 200.0,
                    80.0, 90.0, 220.0, 100.0, 240.0, 120.0])

DIR = os.path.dirname(os.path.realpath(__file__))+"/../../calibration/Ibsen_0107_Serialnumber_missing/"


def test_ibsen_calibration():
    cal_dict = sort_ibsen_by_int(DIR)
    assert np.array(sorted(cal_dict.keys())).all() == IntTime.all()


def test_subtract_dark():
    cal_dict = sort_ibsen_by_int(DIR)
    REFERENCE = copy.deepcopy(cal_dict[5.0]['reference']['tdata'])
    DARK = copy.deepcopy(cal_dict[5.0]['darkcurrent']['tdata'])
    SINGLE_SPECTRA = copy.deepcopy(cal_dict[5.0]['reference']['tdata'][0])
    REFERENCE_MEAN = np.mean(REFERENCE, axis=0)
    DARK_MEAN = np.mean(DARK, axis=0)
    REFERENCE_CORRECTED = REFERENCE_MEAN - DARK_MEAN
    SINGLE_SPECTRA_CORRECTED = SINGLE_SPECTRA - DARK_MEAN

    cal_dict = subtract_dark(cal_dict)

    assert cal_dict[5.0]['reference']['darkcurrent_corrected'] == True
    assert_array_equal(cal_dict[5.0]['reference']['mean'], REFERENCE_CORRECTED)
    assert_array_equal(cal_dict[5.0]['reference']['tdata'][0], SINGLE_SPECTRA_CORRECTED)
    assert_array_equal(cal_dict[5.0]['reference']['data'][:,0], SINGLE_SPECTRA_CORRECTED)


#FILES SHOULD MATCH
#ibsen_response.dat  nonlinearity_gesamt.dat
def prepare_data():
    corrected_calc = subtract_dark(sort_ibsen_by_int(DIR))
    return corrected_calc


def test_generate_nonlinearity_correction():
    corrected_calc = prepare_data()
    nonlinear_config = {'max_lowest_int_time': 1050, 'sigma': 10, 'index_start_spline_fit': 500,
                        'gaussian_mean_steps': 4}
    result_dict = generate_nonlinear_correction(corrected_calc, nonlinear_config)
    DEFAULT_nonlinearity = np.genfromtxt('nonlinearity_gesamt.dat')
    DN_ = DEFAULT_nonlinearity[:,0]
    VALUES = DEFAULT_nonlinearity[:,1]
    assert_array_equal(result_dict['nonlinear'], VALUES)


def test_generate_response():
    corrected_calc = prepare_data()
    cal_dict, response_dict = generate_response_factors(corrected_calc)
    reference_file = '/users/jana_jo/DLR/Codes/calibration/GS1032_1m.txt'
    assert_array_equal(response_dict['halogen'], response_dict['intensity'] / response_dict['scale_factors'])
