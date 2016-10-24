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


def retrieve_aengstrom_parameters(spectra, irr_model, config):
    initial_values = config['initial_values']
    wave_range = config['range_']
    param_dict = dict()
    x_cut, y_cut, std_cut = cut_range(wave_range, spectra['wave_mu'], spectra['spectra'], spectra['std'])
    param_dict['wave_range'] = x_cut
    param_dict['spectra_range'] = y_cut
    param_dict['std'] = std_cut

    # WRONG construct weights with variance
    # cutting new method
    weights = construct_weights(param_dict['std'])
    gmod = Model(irr_model.irradiance_ratio, independent_vars=['x'], param_names=['alpha', 'beta'])
    result = gmod.fit(param_dict['spectra_range'], x=param_dict['wave_range'], alpha=initial_values[0], beta=initial_values[1], weights=weights)

    for key in result.params.keys():
        param_dict[key] = dict()
        param_dict[key]['stderr'] = result.params[key].stderr
        param_dict[key]['value'] = result.params[key].value
    return param_dict, result


class Aerosol_Retrievel(object):

    def __init__(self, irr_model, config, spectra):
        self.spectra = spectra
        self.irr_model = irr_model
        self.fit_range = config['range_']
        self.params = config['params']
        self.initial_values = config['initial_values']
        self.limits = config['limits']
        self.fit_model = None
        self.weights = None

        self.result = None
        self.param_dict = dict()

    def __str__(self):
        var = vars(self)
        return ', '.join("%s: %s" % item for item in var.items())

    def _cut_range(self):
        start = np.where(self.spectra['wave_mu'] <= self.fit_range[0])[0][-1]
        end = np.where(self.spectra['wave_mu'] >= self.fit_range[1])[0][0]
        self.param_dict['wave_range'] = self.spectra['wave_mu'][start:end]
        self.param_dict['spectra_range'] = self.spectra['spectra'][start:end]
        self.param_dict['std'] = self.spectra['std'][start:end]

    def _construct_weights(self):
        self.weights = 1 / self.param_dict['std']

    def set_params(self):
        for param, ini, limits in zip(self.params, self.initial_values, self.limits):
            self.fit_model.set_param_hint(param, value=ini, min=limits[0], max=limits[1])

    def fit(self):
        self._cut_range()
        self._construct_weights()
        self.fit_model = Model(self.irr_model.irradiance_ratio, independent_vars=['x'], param_names=self.params)
        self.set_params()
        self.result = self.fit_model.fit(self.param_dict['spectra_range'], x=self.param_dict['wave_range'], weights=self.weights)

        for key in self.result.params.keys():
            self.param_dict[key] = dict()
            self.param_dict[key]['stderr'] = self.result.params[key].stderr
            self.param_dict[key]['value'] = self.result.params[key].value

def example():
    # possible values
    from get_ssa import get_ssa
    spectra = dict()
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 1020
    AM_type = 6
    ssa = get_ssa(rel_h, AM_type)

    irr = irradiance_models(AMass, rel_h, ssa, zenith, pressure)
    spectra['wave_mu'] = np.linspace(0.2, 0.8, 100)
    spectra['spectra'] = irr.irradiance_ratio(spectra['wave_mu'], 1.2, 0.06) + np.random.normal(0, 0.005, len(spectra['wave_mu']))
    spectra['std'] = np.random.normal(0, 0.1, len(spectra['wave_mu']))
    weights = 1 / spectra['std']

    config_fitting = {'params': np.array(['alpha', 'beta', 'g_dsa', 'g_dsr']), \
                      'initial_values': np.array([ 1. ,  0.6,  1. ,  1. ]), 'range_': np.array([ 0.36,  0.65])}

    aero = Aerosol_Retrievel(irr, config_fitting, spectra)
    aero.fit()
    print(aero.result.fit_report())
    getattr(aero, 'params')

    plt.plot(spectra['wave_mu'], spectra['spectra'])
    plt.plot(aero.param_dict['wave_range'], aero.result.init_fit, 'k--')
    plt.plot(aero.param_dict['wave_range'], aero.result.best_fit, 'r-')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    example()
