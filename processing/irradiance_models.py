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
from utils.util import international_barometric_formula

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
    pressure = international_barometric_formula(config_data['gps_coords'][-1])  # height (magic number)
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
        self.lambda_reference = 550  # [nm] Gege, 1000 nm Greg and Carder + Bringfried

    def forward_scat(self, alpha):
        """
        Reference: Greg and Carder
        empirical values
        """
        if alpha > 1.2:
            cos_theta = 0.65
        else:
            cos_theta = -0.1417 * alpha + 0.82
        B_3 = log(1 - cos_theta)
        B_2 = B_3 * (0.0783 + B_3 * (-0.3824 - 0.5874 * B_3))
        B_1 = B_3 * (1.459 + B_3 * (0.1595 + 0.4129 * B_3))
        F_a = 1 - 0.5 * exp_v((B_1 + B_2 * np.cos(self.zenith_rad)) * np.cos(self.zenith_rad))
        return F_a

    def tau_r(self, x):
        #x_mu = x / 1000
        #return - self.ray * (self. AM * self.pressure / self.p_0) * (x_mu) ** (self.ray_expo)
        y = - ( self.AM * self.pressure/self.p_0) / (115.640 * (x/1000) ** 4 - 1.335 * (x/1000)**2)
        return y

    def tau_as(self, x, alpha, beta):
        return - self.ssa * self.AM * beta * (x / self.lambda_reference)  ** (-alpha)

    def sky_radiance(self, x, alpha, beta, g_dd=1, g_dsa=1, g_dsr=1):
        term = g_dsr * (1 - exp_v(0.95 * self.tau_r(x))) * 0.5 + \
               g_dsa * exp_v(1.5 * self.tau_r(x)) * (1 - exp_v(self.tau_as(x, alpha, beta))) * self.forward_scat(alpha) +  \
               g_dd * exp_v(self.tau_as(x, alpha, beta) + self.tau_r(x))
        return term

    def irradiance_ratio(self, x, alpha, beta, g_dd=0, g_dsa=1, g_dsr=1):
        term = self.sky_radiance(x, alpha, beta, g_dd, g_dsa, g_dsr)
        nom = self.sky_radiance(x, alpha, beta)
        return term / nom


def example():
    # possible values
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 950
    ssas = np.array([])
    AM = 5
    iteration = 20
    alphas = np.zeros(len(range(1,iteration))+1)
    for i in range(0,iteration):
        ssa = get_ssa(rel_h, AM)
        print(ssa)
        irr = irradiance_models(AMass, rel_h, ssa, zenith, pressure)
        x = np.linspace(200, 800, 100)
        y = irr.irradiance_ratio(x, 1.5, 0.06) + np.random.normal(0, 0.009, len(x))
        yerror = np.random.normal(0, 0.009, len(x))
        weights = 1 / yerror

        gmod = Model(irr.irradiance_ratio, independent_vars=['x'], param_names=['alpha', 'beta','g_dsa','g_dsr'])

        gmod.set_param_hint('alpha', value=1.0, min=-0.2)
        gmod.set_param_hint('beta', value=0.01)
        gmod.set_param_hint('g_dsa', value=1., min=0.5, max=1.)
        gmod.set_param_hint('g_dsr', value=1., min=0.5, max=1.)
        print(gmod.param_hints)
        print(gmod.param_names)
        print(gmod.independent_vars)

        result = gmod.fit(y, x=x)
        print(result.fit_report())
        alphas[i] = result.params['alpha'].value
        plt.plot(x, y, label='%s' % AM)
        plt.plot(x, result.best_fit, 'r-')
    plt.legend()
    plt.show()
    print(alphas)
    print(np.mean(alphas))
    print(np.std(alphas))


if __name__ == "__main__":
    example()
