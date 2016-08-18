import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt
from irradiance_models import irradiance_models


def get_spectral_irradiance_reflectance(E_d, E_up):
    return E_up / E_d


def retrieve_aengstrom_parameters(data, irr_model):

    gmod = Model(irr_model.ratio_E_ds_E_d, independent_vars=['x'], param_names=['alpha', 'beta'])
    #pars = gmod.make_params(alpha, beta)
    #result.params['alpha'].stderr
    result = gmod.fit(data['reflect'], x=data['wave_mu'], alpha=0.6, beta=0.21)

    param_dict = dict()
    for key in result.params.keys():
        param_dict[key] = dict()
        param_dict[key]['stderr'] = result.params[key].stderr
        param_dict[key]['value'] = result.params[key].value
    return param_dict, result

