import numpy as np
from processing.irradiance_models import irradiance_models
from processing.spectrum_analyser import retrieve_aengstrom_parameters


def test_retrieve_aengstrom_parameters():
    from numpy.testing import assert_almost_equal
    AM = 1.66450160404
    rel_h = 0.665
    ssa = 0.968997161171
    zenith = 20

    irr = irradiance_models(AM, rel_h, ssa, zenith)
    x = np.linspace(200, 800, 100) / 1000.
    y = irr.ratio_E_ds_E_d(x, 1.2, 0.06) + np.random.normal(0, 0.01, len(x))

    reflectance_dict = {'wave_mu': x, 'reflect': y}
    w_range = np.array([200., 800.]) / 1000.
    params, result = retrieve_aengstrom_parameters(reflectance_dict, irr, w_range)
    assert_almost_equal(params['alpha']['value'], 1.2, 1)
