import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt
from irradiance_models import irradiance_models
from utils.util import construct_weights


def get_spectral_irradiance_reflectance(E_d, E_up):
    return E_up / E_d


def get_error_of_spectral_reflectance(E_d, E_up, E_d_std, E_up_std):
    """
    Reflectance std is calculated via gaussian propagation of uncertainty.
    Specific error is calculated
    """
    reflect_std = np.sqrt((1 / E_d) ** 2 * E_up_std ** 2 + (E_up / E_d ** 2) ** 2 * E_d_std ** 2)
    return reflect_std

def retrieve_aengstrom_parameters(spectra, irr_model, wave_range, initial_values):
    """
    Args:
        spectra: Dict with relflectance and wavelength [\mum]
        irr_model: Irradiance_model object
        wave_range: Wave range to fit
        inital_value: Dict with inital values with parameter name keys
    """

    #pars = gmod.make_params(alpha, beta)
    #result.params['alpha'].stderr
    weights = construct_weights(spectra['wave_mu'], wave_range)
    gmod = Model(irr_model.ratio_E_ds_E_d, independent_vars=['x'], param_names=['alpha', 'beta'])
    result = gmod.fit(spectra['spectra'], x=spectra['wave_mu'], alpha=initial_values['alpha'], beta=initial_values['beta'])

    param_dict = dict()
    for key in result.params.keys():
        param_dict[key] = dict()
        param_dict[key]['stderr'] = result.params[key].stderr
        param_dict[key]['value'] = result.params[key].value
    return param_dict, result

