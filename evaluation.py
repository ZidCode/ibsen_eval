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
from parser.ibsen_parser import parse_ibsen_file, get_mean_column, get_mean_column
from utils.plotting import plot_meas, plot_used_irradiance_and_reflectance, plot_fitted_reflectance, plot_aengstrom_parameters
from calibration.ibsen_calibration import subtract_dark_from_mean
from parser.microtops import extract_microtops_inifile
"""
    Irradiance measurements does not show calibrated spectra
    Fit class necessary
"""


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
    config_dict['Validation']['validate'] = literal_eval(config_dict['Validation']['validate'])
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
def evaluate_spectra(config, logger=logging):
    ref = parse_ibsen_file(config['Data']['reference'])
    tar = parse_ibsen_file(config['Data']['target'])
    dark = parse_ibsen_file(config['Data']['darkcurrent'])
    subtract_dark_from_mean(dark, tar)
    subtract_dark_from_mean(dark, ref)
    if tar['UTCTime']:
        logger.warning("Config UTCTime: %s. New UTCTime %s from IbsenFile." % (config['Processing']['utc_time'], tar['UTCTime']))
        config['Processing']['utc_time'] = tar['UTCTime']
    logger.info("Date: %s " % config['Processing']['utc_time'])
    logger.info("GPS coords (lat, lon) %s %s" % (config['Processing']['gps_coords'][0], config['Processing']['gps_coords'][1]))
    logger.info("Files\n \t ref: %s  \n \t tar: %s \n \t dark: %s" %(config['Data']['reference'], config['Data']['target'], config['Data']['darkcurrent'] ))

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


def evaluate_measurements(directory, config, logger=logging):
    import glob
    file_prefixes = ['target', 'reference', 'darkcurrent']
    files = [file_ for file_ in glob.iglob(directory + '%s*' % file_prefixes[0])]
    keys_for_param_dict = ['utc_times', 'alpha', 'alpha_stderr', 'beta', 'beta_stderr']
    param_dict = {key:np.array([]) for key in keys_for_param_dict}
    param_dict['label'] = 'Ibsen'
    for file_ in sorted(files):
        for key in file_prefixes:
            config['Data'][key] = file_.replace(file_prefixes[0], key)
        try:
            logger.info("Evaluating file: %s \n" % file_)
            params = evaluate_spectra(config)
            param_dict['utc_times'] = np.append(param_dict['utc_times'], config['Processing']['utc_time'])
            param_dict['alpha'] = np.append(param_dict['alpha'], params['alpha']['value'])
            param_dict['alpha_stderr'] = np.append(param_dict['alpha_stderr'], params['alpha']['stderr'])
            param_dict['beta'] = np.append(param_dict['beta'], params['beta']['value'])
            param_dict['beta_stderr'] = np.append(param_dict['beta_stderr'], params['beta']['stderr'])
        except IOError:
            logger.error("%s have no corresponding reference or darkcurrentfiles" % file_)

    # Microtops
    if config['Validation']['validate']:
        validation = extract_microtops_inifile(config['Validation'], config['Processing']['utc_time'])
        plot_aengstrom_parameters(param_dict, validation)
    else:
        plot_aengstrom_parameters(param_dict)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='Pass ini-file for processing configurations')
    parser.add_argument('-m', '--measurement_directory', help='Define measurement directory to sweep through')
    args = parser.parse_args()
    config = parse_ini_config(args.config)
    logger = create_logger(config['Processing'])
    if args.measurement_directory:
        evaluate_measurements(args.measurement_directory, config, logger)
    else:
        evaluate_spectra(config, logger)
