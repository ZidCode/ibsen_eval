import logging
import numpy as np
from scipy.constants import atmosphere
from math import exp, log
from lmfit import Model
import matplotlib.pyplot as plt
from solar_zenith import get_sun_zenith
from get_weather_conditions import retrieve_weather_parameters
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
    weather_dict = retrieve_weather_parameters(config_data['params'], config_data['gps_coords'], config_data['utc_time'])
    humidity = weather_dict['hum']
    pressure = weather_dict['pressurem']
    ssa = get_ssa(humidity)
    irr_mod = irradiance_models(atmos_path, humidity, ssa, sun_zenith, pressure)
    logger.info(" \n \t Zenith angle %s" %  sun_zenith)
    logger.info(" \n \t  Atmospheric path length %s" % atmos_path)
    logger.info(" \n \t  Relative humidity %s" % humidity)
    logger.info(" \n \t  Pressure %s" % pressure)
    logger.info(" \n \t  Single scattering albedo %s" % ssa)
    return irr_mod


class irradiance_models:

    def __init__(self, AM, RH, ssa, zenith, pressure):
        self.AM = AM
        self.RH = RH
        self.ssa = ssa
        self.pressure = pressure
        self.p_0 = atmosphere / 100.0
        self.zenith_rad = np.radians(zenith)
        self.ray = 0.00877
        self.ray_expo = -4.05

    def forward_scat(self, alpha):
        """
        Reference: Greg and Carder
        empirical values
        """
        cos_theta = -0.1417 * alpha + 0.82
        B_3 = log(1 - cos_theta)
        B_2 = B_3 * (0.0783 + B_3 * (-0.3824 - 0.5874 * B_3))
        B_1 = B_3 * (1.459 + B_3 * (0.1595 + 0.4129 * B_3))
        F_a = 1 - 0.5 * exp_v((B_1 + B_2 * np.cos(self.zenith_rad)) * np.cos(self.zenith_rad))
        return F_a

    def tau_r(self, x):
        return - self.ray * (self.pressure / self.p_0) * x ** (self.ray_expo)

    def tau_as(self, x, alpha, beta):
        return - self.ssa * self.AM * beta * x ** (-alpha)

    def mix_term(self, x, alpha, beta):
        term = (1 - exp_v(0.95 * self.tau_r(x)))  * 0.5 + exp_v(1.5 * self.tau_r(x)) * (1 - exp_v(self.tau_as(x, alpha, beta)))* self.forward_scat(alpha)
        return term

    def ratio_E_ds_E_d(self, x, alpha, beta):
        term = self.mix_term(x, alpha, beta)
        nom = self.mix_term(x, alpha, beta) + exp_v(self.tau_as(x, alpha, beta) + self.tau_r(x))
        return term / nom


def example():
    # possible values
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AM = 1.66450160404
    rel_h = 0.665
    pressure = 1020
    ssas = np.array([])
    AM_type = np.arange(1,11)
    for AM in AM_type:

        ssa = get_ssa(rel_h, AM)
        irr = irradiance_models(AM, rel_h, ssa, zenith, pressure)
        x = np.linspace(0.2, 0.8, 100)
        y = irr.ratio_E_ds_E_d(x, 1.2, 0.06) + np.random.normal(0, 0.01, len(x))
        yerror = np.random.normal(0, 0.1, len(x))
        weights = 1 / yerror

        gmod = Model(irr.ratio_E_ds_E_d, independent_vars=['x'], param_names=['alpha', 'beta'])
        print(gmod.param_names)
        print(gmod.independent_vars)

        result = gmod.fit(y, x=x, alpha=0.6, beta=0.21, weights=weights)
        print(result.fit_report())

        plt.plot(x, y, label='%s' % AM)
        plt.plot(x, result.init_fit, 'k--')
        plt.plot(x, result.best_fit, 'r-')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    example()
