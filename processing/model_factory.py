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
from Model import IrradianceRatio
from BaseModels import BaseModelPython
"""
Irradiance model due to Greg and Carder
"""


class WeatherAtmosphereParameter:

    def __init__(self, logger, config, wavelength):
        self.sun_zenith_tar = get_sun_zenith(config['Processing']['utc_time']['tar'], *config['Processing']['gps_coords'])
        self.sun_zenith_ref = get_sun_zenith(config['Processing']['utc_time']['ref'], *config['Processing']['gps_coords'])
        self.atmos_path = get_atmospheric_path_length(self.sun_zenith_tar)
        weather_dict = retrieve_weather_parameters(config['Processing']['params'], config['Processing']['gps_coords'], config['Processing']['utc_time']['tar'])
        humidity = weather_dict['hum']
        self.pressure = international_barometric_formula(config['Processing']['gps_coords'][-1])  # height (magic number)
        self.ssa = get_ssa(humidity)
        logger.info(" \n \t Zenith angle tar %s" %  self.sun_zenith_tar)
        logger.info(" \n \t Zenith angle ref %s" %  self.sun_zenith_ref)
        logger.info(" \n \t  Atmospheric path length tar %s" % self.atmos_path)
        logger.info(" \n \t  Atmospheric path length ref %s" % get_atmospheric_path_length(self.sun_zenith_ref))
        logger.info(" \n \t  Relative humidity %s" % humidity)
        logger.info(" \n \t  Pressure %s" % self.pressure)
        logger.info(" \n \t  Single scattering albedo %s" % self.ssa)
        logger.info(" \n \t  Using %s model <<<< FIT" % config['Fitting']['package'] )



def example():
    # possible values
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 950
    AM = 5
    ssa = get_ssa(rel_h, AM)
    iteration = 20
    alphas = np.zeros(len(range(1,iteration))+1)

    x = np.linspace(200, 800, 100)
    irr = BaseModelPython(AMass, rel_h, ssa, zenith, pressure)
    irr_symbol = BaseModelPython(x, zenith, AMass, pressure, ssa)

    func = irr_symbol._irradiance_ratio()

    y = irr.irradiance_ratio(x, 2.5, 0.06, 0.0, 1., 1.)
    for i in range(0,iteration):
        ssa = get_ssa(rel_h, AM)
        print(ssa)
        irr = BaseModelPython(AMass, rel_h, ssa, zenith, pressure)
        yerror = np.random.normal(0, 0.009, len(x))
        y = irr.irradiance_ratio(x, 1.5, 0.06, 0.0, 0.6, 0.9) + yerror
        weights = 1 / yerror

        gmod = Model(irr.irradiance_ratio, independent_vars=['x'], param_names=['alpha', 'beta','g_dsa','g_dsr'])

        gmod.set_param_hint('alpha', value=1.0, min=-0.2, max=2.5)
        gmod.set_param_hint('beta', value=0.01, min=0.0, max = 2.)
        gmod.set_param_hint('g_dsa', value=0.6, min=0., max=1.)
        gmod.set_param_hint('g_dsr', value=0.9, min=0., max=1.)
        print(gmod.param_hints)
        print(gmod.param_names)
        print(gmod.independent_vars)

        result = gmod.fit(y, x=x)
        print(result.fit_report())
        alphas[i] = result.params['alpha'].value

        # plt.plot(x, y, label='%s' % AM)
        # plt.plot(x, result.best_fit, 'r-', label='fit')
    y = irr.irradiance_ratio(x, 1.5, 0.06, 0.0, 0.6, 0.9)
    y2 = irr.irradiance_ratio(x, 1.5, 0.08, 0.0, 0.6, 0.9)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    example()
