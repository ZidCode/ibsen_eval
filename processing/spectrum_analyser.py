import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt
from irradiance_models import irradiance_models
from utils.util import construct_weights


def get_spectral_irradiance_reflectance(E_d, E_up):
    return E_up / E_d


def retrieve_aengstrom_parameters(spectra, irr_model, wave_range):

    #pars = gmod.make_params(alpha, beta)
    #result.params['alpha'].stderr
    weights = construct_weights(spectra['wave_mu'], wave_range)
    gmod = Model(irr_model.ratio_E_ds_E_d, independent_vars=['x'], param_names=['alpha', 'beta'])
    result = gmod.fit(spectra['reflect'], x=spectra['wave_mu'], alpha=1.4, beta=0.0, weights=weights)

    param_dict = dict()
    for key in result.params.keys():
        param_dict[key] = dict()
        param_dict[key]['stderr'] = result.params[key].stderr
        param_dict[key]['value'] = result.params[key].value
    return param_dict, result

