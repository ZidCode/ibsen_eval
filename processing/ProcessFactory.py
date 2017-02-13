import numpy as np
from spektralon.spektralon import scale_to_irradiance
import logging


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


class WrongModelException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = "No %s model impolemented" % msg


class DataProcess:
    def __init__(self, model, logger=logging):
        self.model = model
        self.logger = logger

    def __call__(self):
        if self.model == 'ratio':
            self.logger.info("E_ds_E_d Ratio Process")
            return Ratio()
        elif self.model == 'l_sky_ratio':
            self.logger.info("L_sky_E_d Ratio Process")
            return LSkyRatio()
        elif self.model == 'l_sky_nadir':
            self.logger.info("L_sky_nadir Process")
            return LSky()
        else:
            raise WrongModelException(self.model)


class LSky:
    def __init__(self):
        pass

    def process(self, __, target):
        data_dict = {'wave_nm': target['wave'], 'spectra': target['mean'], 'std': target['data_sample_std'], 'var': target['data_sample_std'] ** 2}
        return data_dict


class Ratio:
    def __init__(self):
        pass

    def process(self, reference, target):
        return get_reflectance(reference, target)


class LSkyRatio:
    def __init__(self):
        pass

    def process(self, reference, target):
        ref = scale_to_irradiance(reference)
        return get_reflectance(ref, target)
