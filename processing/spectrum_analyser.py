import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt
from irradiance_models import irradiance_models
from utils.util import construct_weights, cut_range


def get_spectral_irradiance_reflectance(E_d, E_up):
    return E_up / E_d


def get_error_of_spectral_reflectance(E_d, E_up, E_d_std, E_up_std):
    """
    Reflectance std is calculated via gaussian propagation of uncertainty.
    Specific error is calculated
    """
    reflect_std = np.sqrt((1 / E_d) ** 2 * E_up_std ** 2 + (E_up / E_d ** 2) ** 2 * E_d_std ** 2)
    return reflect_std

def get_reflectance(E_d, E_up):
    reflectance = get_spectral_irradiance_reflectance(E_d['mean'], E_up['mean'])
    reflectance_std = get_error_of_spectral_reflectance(E_d['mean'], E_up['mean'], E_d['data_sample_std'], E_up['data_sample_std'])

    reflectance_dict = {'wave_mu': E_d['wave'] / 1000.0, 'spectra': reflectance, 'std': reflectance_std, 'var': reflectance_std ** 2}
    return reflectance_dict


def retrieve_aengstrom_parameters(spectra, irr_model, wave_range, initial_values):
    """
    Args:
        spectra: Dict with relflectance and wavelength [\mum]
        irr_model: Irradiance_model object
        wave_range: Wave range to fit
        inital_value: Dict with inital values with parameter name keys
    """
    param_dict = dict()
    x_cut, y_cut, std_cut = cut_range(wave_range, spectra['wave_mu'], spectra['spectra'], spectra['std'])
    param_dict['wave_range'] = x_cut
    param_dict['spectra_range'] = y_cut
    param_dict['std'] = std_cut

    # WRONG construct weights with variance
    # cutting new method
    weights = construct_weights(param_dict['std'])
    gmod = Model(irr_model.ratio_E_ds_E_d, independent_vars=['x'], param_names=['alpha', 'beta'])
    result = gmod.fit(param_dict['spectra_range'], x=param_dict['wave_range'], alpha=initial_values['alpha'], beta=initial_values['beta'], weights=weights)

    for key in result.params.keys():
        param_dict[key] = dict()
        param_dict[key]['stderr'] = result.params[key].stderr
        param_dict[key]['value'] = result.params[key].value
    return param_dict, result

