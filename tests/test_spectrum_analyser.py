import numpy as np
from numpy import allclose
from processing.irradiance_models import irradiance_models
from processing.spectrum_analyser import get_error_of_spectral_reflectance


def test_get_error_of_spectral_reflectance():
    tar = np.array([50., 70.])
    ref = np.array([200., 300.])
    ref_std = np.array([0.3, 0.2])
    tar_std = np.array([0.2, 0.5])
    REF_STD = np.array([ 0.001068  ,  0.00167391])
    ref_std = get_error_of_spectral_reflectance(ref, tar, ref_std, tar_std)
    allclose(REF_STD, ref_std)
