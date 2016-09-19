#!/usr/bin/env python
import os
import re
import copy
import logging
import ConfigParser
import numpy as np
import pandas as pd
from datetime import datetime
from ast import literal_eval
import processing.irradiance_models as irr
from processing.spectrum_analyser import get_reflectance, retrieve_aengstrom_parameters
from parser.ibsen_parser import parse_ibsen_file, get_mean_column, get_mean_column, subtract_dark_from_mean
from utils.plotting import plot_meas, plot_used_irradiance_and_reflectance, plot_fitted_reflectance


def parse_ini_config(ini_file):
    config = ConfigParser.ConfigParser()
    config.read(ini_file)
    config_dict = {s: dict(config.items(s)) for s in config.sections()}
    config_dict['Processing']['logging'] = literal_eval(config_dict['Processing']['logging'])
    config_dict['Processing']['gps_coords'] = np.array([float(s) for s in config_dict['Processing']['gps_coords'].split(',')])
    config_dict['Processing']['utc_time'] = datetime.strptime(config_dict['Processing']['utc_time'], '%Y-%m-%d %H:%M:%S')
    config_dict['Processing']['params'] = re.split(', | \s', config_dict['Processing']['params'])
    config_dict['Fitting']['range_'] = np.array([float(s) for s in config_dict['Fitting']['range_'].split(',')])
    config_dict['Fitting']['alpha'] = float(config_dict['Fitting']['alpha'])
    config_dict['Fitting']['beta'] = float(config_dict['Fitting']['beta'])
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
def evaluate_spectra(config):
    logger = create_logger(config['Processing'])
    ref = parse_ibsen_file(config['Data']['reference'])
    tar = parse_ibsen_file(config['Data']['target'])
    dark = parse_ibsen_file(config['Data']['dark'])
    subtract_dark_from_mean(dark, tar, ref)

    if tar['UTCTime']:
        logger.warning("Config UTCTime: %s. New UTCTime %s from IbsenFile." % (config['Processing']['utc_time'], tar['UTCTime']))
        config['Processing']['utc_time'] = tar['UTCTime']
    logger.info("Date: %s " % config['Processing']['utc_time'])
    logger.info("GPS coords (lat, lon) %s %s" % (config['Processing']['gps_coords'][0], config['Processing']['gps_coords'][1]))
    logger.info("Files\n \t ref: %s  \n \t tar: %s \n \t dark: %s" %(config['Data']['reference'], config['Data']['target'], config['Data']['dark'] ))

    irradiance_model = irr.build_Model(config['Processing'], logger)
    reflectance_dict = get_reflectance(ref, tar)

    inital_values = config['Fitting']
    range_ = config['Fitting']['range_']

    params, result = retrieve_aengstrom_parameters(reflectance_dict, irradiance_model, range_, inital_values)

    logger.info("%s \n" % result.fit_report())


    if config['Processing']['logging_level'] == 'DEBUG':
        plot_meas(tar, ref, dark)
        #frame = pd.DataFrame(np.transpose([tar['wave'], reflectance_dict['spectra']]), columns=['Wavelength', 'Reflectance'])
        #frame.to_csv('reflectance.csv', index=False)
        plot_used_irradiance_and_reflectance(tar, ref, reflectance_dict)
        plot_fitted_reflectance(reflectance_dict, params, result)

    return params

if __name__ == "__main__":
    default_ini = 'config.ini'
    config = parse_ini_config(default_ini)
    evaluate_spectra(config)
