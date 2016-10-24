import numpy as np
from numpy import allclose
from processing.irradiance_models import irradiance_models
from processing.spectrum_analyser import retrieve_aengstrom_parameters, get_error_of_spectral_reflectance


def test_retrieve_aengstrom_parameters():
    from numpy.testing import assert_almost_equal
    AM = 1.66450160404
    rel_h = 0.665
    ssa = 0.968997161171
    zenith = 20
    pressure = 1021
    config = dict()

    irr = irradiance_models(AM, rel_h, ssa, zenith, pressure)
    x = np.linspace(200, 800, 100) / 1000.
    y = irr.irradiance_ratio(x, 1.2, 0.06)

    reflectance_dict = {'wave_mu': x, 'spectra': y, 'std': np.random.normal(0, 0.01, len(y))}
    config['range_'] = np.array([200., 800.]) / 1000.
    config['initial_values'] = [1.2, 0.05]
    params, result = retrieve_aengstrom_parameters(reflectance_dict, irr, config)
    assert_almost_equal(params['alpha']['value'], 1.2, 1)


def test_get_error_of_spectral_reflectance():
    tar = np.array([50., 70.])
    ref = np.array([200., 300.])
    ref_std = np.array([0.3, 0.2])
    tar_std = np.array([0.2, 0.5])
    REF_STD = np.array([ 0.001068  ,  0.00167391])
    ref_std = get_error_of_spectral_reflectance(ref, tar, ref_std, tar_std)
    allclose(REF_STD, ref_std)
