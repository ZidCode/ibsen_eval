#!/usr/bin/env python
import os
import copy
import logging
import ConfigParser
import numpy as np
import pandas as pd
from datetime import datetime
from ast import literal_eval
import processing.irradiance_models as irr
from processing.spectrum_analyser import get_spectral_irradiance_reflectance, retrieve_aengstrom_parameters
from parser.ibsen_parser import parse_ibsen_file, get_mean_column, get_mean_column, subtract_dark_from_mean
from utils.plotting import plot_meas, plot_used_irradiance_and_reflectance


def parse_ini_config(ini_file):
    config = ConfigParser.ConfigParser()
    config.read(ini_file)
    config_dict = {s: dict(config.items(s)) for s in config.sections()}
    config_dict['Processing']['logging'] = literal_eval(config_dict['Processing']['logging'])
    config_dict['Data']['gps_coords'] = np.array([float(s) for s in config_dict['Data']['gps_coords'].split(',')])
    config_dict['Data']['utc_time'] = datetime.strptime(config_dict['Data']['utc_time'], '%Y-%m-%d %H:%M:%S')
    config_dict['Processing']['range_'] = np.array([float(s) for s in config_dict['Processing']['range_'].split(',')])
    config_dict['Processing']['alpha'] = float(config_dict['Processing']['alpha'])
    config_dict['Processing']['beta'] = float(config_dict['Processing']['beta'])
    return config_dict


def create_logger(log_config):
    log_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO}
    logger = logging.getLogger('eval')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(log_config['logging_level'])
    return logger


# Decorators
def evaluate(config):
    logger = create_logger(config['Processing'])
    ref = parse_ibsen_file(config['Data']['reference'])
    tar = parse_ibsen_file(config['Data']['target'])
    dark = parse_ibsen_file(config['Data']['dark'])
    subtract_dark_from_mean(dark, tar, ref)

    logger.info("Date: %s " % config['Data']['utc_time'])
    logger.info("GPS coords (lat, lon) %s %s" % (config['Data']['gps_coords'][0], config['Data']['gps_coords'][1]))
    logger.info("Files\n \t ref: %s  \n \t tar: %s \n \t dark: %s" %(config['Data']['reference'], config['Data']['target'], config['Data']['dark'] ))

    reflectance = get_spectral_irradiance_reflectance(ref['mean'], tar['mean'])
    irradiance_model = irr.build_Model(config['Data'], logger)
    reflectance_dict = {'wave_mu': ref['wave'] / 1000.0, 'reflect': reflectance}

    inital_values = config['Processing']
    range_ = config['Processing']['range_']
    params, result = retrieve_aengstrom_parameters(reflectance_dict, irradiance_model, range_, inital_values)
    logger.info("%s \n" % result.fit_report())

    import matplotlib.pyplot as plt
    plt.plot(reflectance_dict['wave_mu'], result.init_fit, 'k--')
    plt.plot(reflectance_dict['wave_mu'], result.best_fit, 'r-')
    plt.plot(reflectance_dict['wave_mu'], reflectance, '*')
    plt.show()


    if config['Processing']['logging_level'] == 'DEBUG':
        plot_meas(tar, ref, dark)
        frame = pd.DataFrame(np.transpose([tar['wave'], reflectance_dict['spectra']]), columns=['Wavelength', 'Reflectance'])
        frame.to_csv('reflectance.csv', index=False)
        plot_used_irradiance_and_reflectance(tar, ref, reflectance_dict)


if __name__ == "__main__":
    default_ini = 'config.ini'
    config = parse_ini_config(default_ini)
    evaluate(config)
