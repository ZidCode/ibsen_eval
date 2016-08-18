import logging
import numpy as np
from math import exp
from lmfit import Model
import matplotlib.pyplot as plt
from solar_zenith import get_sun_zenith
from get_weather_conditions import retrieve_rel_humidity
from atmospheric_mass import get_atmospheric_path_length
from get_ssa import get_ssa

"""
Irradiance model due to Greg and Carder
"""
exp_v = np.vectorize(exp)


def build_Model(config_data, logger=logging):
    """
    This method constructs a class object with calculated parameters from gps
    position and utc_time
    Args:
        config_data: Dict with gps_coords (list of floats) and utc_time
        (datetime)
        logger: available logger
    Returns:
        irr_mod: irradiance_modles object
    """
    sun_zenith = get_sun_zenith(config_data['utc_time'], *config_data['gps_coords'])
    atmos_path = get_atmospheric_path_length(sun_zenith)
    RH = retrieve_rel_humidity(config_data['gps_coords'], config_data['utc_time'])
    ssa = get_ssa(RH)
    irr_mod = irradiance_models(atmos_path, RH, ssa)
    logger.info(" \n \t Zenith angle %s" %  sun_zenith)
    logger.info(" \n \t  Atmospheric path length %s" % atmos_path)
    logger.info(" \n \t  Relative humidity %s" % RH)
    logger.info(" \n \t  Single scattering albedo %s" % ssa)
    return irr_mod


class irradiance_models:

    def __init__(self, AM, RH, ssa):
        self.AM = AM
        self.RH = RH
        self.ssa = ssa

        self.ray = 0.00877
        self.ray_expo = -4.05

    def tau_r(self, x):
        return - self.ray * x ** (self.ray_expo)

    def tau_as(self, x, beta, alpha):
        return - self.ssa * self.AM * beta * x ** (-alpha)

    def mix_term(self, x, alpha, beta):
        term = (1 - exp_v(0.95 * self.tau_r(x)))  * 0.5 + exp_v(1.5 * self.tau_r(x)) * (1 - exp_v(self.tau_as(x, beta, alpha)))
        return term

    def ratio_E_ds_E_d(self, x, alpha, beta):
        term = self.mix_term(x, alpha, beta)
        nom = self.mix_term(x, alpha, beta) + exp_v(self.tau_as(x, alpha, beta) + self.tau_r(x))
        return term / nom


def example():
    # possible values
    AM = 1.66450160404
    rel_h = 0.665
    ssa = 0.968997161171

    irr = irradiance_models(AM, rel_h, ssa)
    x = np.linspace(0.2, 0.8, 100)
    y = irr.ratio_E_ds_E_d(x, 1.2, 0.06) + np.random.normal(0, 0.01, len(x))


    gmod = Model(irr.ratio_E_ds_E_d, independent_vars=['x'], param_names=['alpha', 'beta'])
    print(gmod.param_names)
    print(gmod.independent_vars)

    result = gmod.fit(y, x=x, alpha=0.6, beta=0.21)

    print(result.fit_report())
    result.plot()

    plt.plot(x, y,         'bo')
    plt.plot(x, result.init_fit, 'k--')
    plt.plot(x, result.best_fit, 'r-')
    plt.show()


if __name__ == "__main__":
    example()
