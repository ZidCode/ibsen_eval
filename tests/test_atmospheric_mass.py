#!/usr/bin/env python
from evaluation.processing.atmospheric_mass import get_atmospheric_path_length
from numpy.testing import assert_almost_equal


def test_get_atmospheric_path_length():
    # Backchecked with Matlab
    zenith = 40
    AM_matlab = 1.3040
    AM = get_atmospheric_path_length(zenith)
    assert_almost_equal(AM, AM_matlab, 4)
