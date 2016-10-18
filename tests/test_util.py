from numpy.testing import assert_almost_equal
from evaluation.utils.util import international_barometric_formula

def test_international_barometric_formula():
    PRESSURE = 944
    height = 590
    pressure = international_barometric_formula(height)
    assert_almost_equal(PRESSURE, pressure, 0)
