import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt
from irradiance_models import irradiance_models


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

    reflectance_dict = {'wave_nm': E_d['wave'], 'spectra': reflectance, 'std': reflectance_std, 'var': reflectance_std ** 2}
    return reflectance_dict


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
        start = np.where(self.spectra['wave_nm'] <= self.fit_range[0])[0][-1]
        end = np.where(self.spectra['wave_nm'] >= self.fit_range[1])[0][0]
        self.param_dict['wave_range'] = self.spectra['wave_nm'][start:end]
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
    spectra['wave_nm'] = np.linspace(200, 800, 100)
    spectra['spectra'] = irr.irradiance_ratio(spectra['wave_nm'], 1.2, 0.06) + np.random.normal(0, 0.005, len(spectra['wave_nm']))
    spectra['std'] = np.random.normal(0, 0.1, len(spectra['wave_nm']))
    weights = 1 / spectra['std']

    config_fitting = {'params': np.array(['alpha', 'beta', 'g_dsa', 'g_dsr']), \
                      'initial_values': np.array([ 1. ,  0.6,  1. ,  1. ]), 'range_': np.array([360,  650])}

    aero = Aerosol_Retrievel(irr, config_fitting, spectra)
    aero.fit()
    print(aero.result.fit_report())
    getattr(aero, 'params')

    plt.plot(spectra['wave_nm'], spectra['spectra'])
    plt.plot(aero.param_dict['wave_range'], aero.result.init_fit, 'k--')
    plt.plot(aero.param_dict['wave_range'], aero.result.best_fit, 'r-')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    example()
