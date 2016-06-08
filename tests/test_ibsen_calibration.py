import numpy as np
from evaluation.parser.ibsen_calibration import sort_ibsen_by_int, get_halogen_spectra, generate_ibsen_cal


IntTime = np.array([260.0, 5.0, 10.0, 140.0, 15.0, 20.0, 280.0, 25.0, 30.0,
                    160.0, 35.0, 40.0, 45.0, 50.0, 180.0, 60.0, 70.0, 200.0,
                    80.0, 90.0, 220.0, 100.0, 240.0, 120.0])

def test_ibsen_calibration():
    test_dict = generate_ibsen_cal()
    assert np.array(sorted(test_dict.keys())).all() == IntTime.all()


def test_get_halogen_spectra():
    # TODO
    assert True
