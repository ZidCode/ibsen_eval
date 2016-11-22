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
from processing.spectrum_analyser import get_reflectance, Aerosol_Retrievel
from parser.ibsen_parser import parse_ibsen_file, get_mean_column, get_mean_column
from utils.plotting import plot_meas, plot_used_irradiance_and_reflectance, plot_fitted_reflectance, plot_aengstrom_parameters
from parser.microtops import extract_microtops_inifile
"""
    Irradiance measurements does not show calibrated spectra
    Fit class necessary
"""

convert_to_array = lambda x, m : np.array([m(s) for s in x.split(',')])

def parse_ini_config(ini_file):
    config = ConfigParser.ConfigParser()
    config.read(ini_file)
    config_dict = {s: dict(config.items(s)) for s in config.sections()}
    config_dict['Processing']['logging'] = literal_eval(config_dict['Processing']['logging'])
    config_dict['Processing']['gps_coords'] = convert_to_array(config_dict['Processing']['gps_coords'], float)
    config_dict['Processing']['utc_time'] = datetime.strptime(config_dict['Processing']['utc_time'], '%Y-%m-%d %H:%M:%S')
    config_dict['Processing']['params'] = re.split(', | \s', config_dict['Processing']['params'])
    config_dict['Fitting']['limits'] = [convert_to_array(val, float) for val in config_dict['Fitting']['limits'].split(';')]
    config_dict['Fitting']['range_'] = convert_to_array(config_dict['Fitting']['range_'], float)
    config_dict['Fitting']['params']  = convert_to_array(config_dict['Fitting']['params'], str)
    config_dict['Fitting']['initial_values'] = convert_to_array(config_dict['Fitting']['initial_values'], float)
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

    if tar['UTCTime']:
        logger.warning("Config UTCTime: %s. New UTCTime %s from IbsenFile." % (config['Processing']['utc_time'], tar['UTCTime']))
        config['Processing']['utc_time'] = tar['UTCTime']
    logger.info("Date: %s " % config['Processing']['utc_time'])
    logger.info("GPS coords (lat, lon) %s %s" % (config['Processing']['gps_coords'][0], config['Processing']['gps_coords'][1]))
    logger.info("Files\n \t ref: %s  \n \t tar: %s " %(config['Data']['reference'], config['Data']['target']))

    irradiance_model = irr.build_Model(config['Processing'], logger)

    # if config['Processing']['spectralon'] == 'reference':
    #     spektralon = np.genfromtxt('/home/jana_jo/DLR/Codes/calibration/Spektralon/Spectralon_neu.txt', skip_header=12)
    #     ref['mean'] = ref['mean'] / np.interp(ref['wave'], spektralon[:,0], spektralon[:,1])
    reflectance_dict = get_reflectance(ref, tar)

    aero = Aerosol_Retrievel(irradiance_model, config['Fitting'], reflectance_dict)
    aero.fit()
    logger.info("%s \n" % aero.result.fit_report())

    if config['Processing']['logging_level'] == 'DEBUG':
        plot_meas(tar, ref)
        #frame = pd.DataFrame(np.transpose([tar['wave'], reflectance_dict['spectra']]), columns=['Wavelength', 'Reflectance'])
        #frame.to_csv('reflectance.csv', index=False)
        plot_used_irradiance_and_reflectance(tar, ref, reflectance_dict)
        plot_fitted_reflectance(aero)
    return aero.param_dict, aero


def evaluate_measurements(directory, config, logger=logging):
    import glob
    file_prefixes = ['target', 'reference']
    files = [file_ for file_ in glob.iglob(directory + '%s*' % file_prefixes[0])]
    keys_for_param_dict = ['utc_times', 'alpha', 'alpha_stderr', 'beta', 'beta_stderr']
    param_dict = {key:np.array([]) for key in keys_for_param_dict}
    param_dict['label'] = 'Ibsen'


    for file_ in sorted(files):
        for key in file_prefixes:
            config['Data'][key] = file_.replace(file_prefixes[0], key)
        try:
            logger.info("Evaluating file: %s \n" % file_)
            params, aero = evaluate_spectra(config, logger)
            param_dict['utc_times'] = np.append(param_dict['utc_times'], config['Processing']['utc_time'])
            param_dict['alpha'] = np.append(param_dict['alpha'], params['alpha']['value'])
            param_dict['alpha_stderr'] = np.append(param_dict['alpha_stderr'], params['alpha']['stderr'])
            param_dict['beta'] = np.append(param_dict['beta'], params['beta']['value'])
            param_dict['beta_stderr'] = np.append(param_dict['beta_stderr'], params['beta']['stderr'])
        except IOError:
            logger.error("%s have no corresponding reference" % file_)

    # Microtops
    if config['Validation']['validate']:
        validation = extract_microtops_inifile(config['Validation'], config['Processing']['utc_time'])
        plot_aengstrom_parameters(param_dict, validation)
    else:
        plot_aengstrom_parameters(param_dict)

def compare_measurements(config, logger):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec


    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    print(config)
    param_dict, aero_fit = evaluate_spectra(config, logger)
    ax1 = aero_fit.result.plot_residuals(ax=ax1, datafmt='g')
    ax2.plot(aero_fit.spectra['wave_nm'], aero_fit.spectra['spectra'])
    ax2.plot(aero_fit.param_dict['wave_range'], aero_fit.result.best_fit, 'g-')
    ax2.errorbar(aero_fit.param_dict['wave_range'], aero_fit.param_dict['spectra_range'], yerr=aero_fit.param_dict['std'], ecolor='g')

    config['Fitting'] = {'params': np.array(['alpha', 'beta', 'g_dsa', 'g_dsr'], dtype='|S5'), 'initial_values': np.array([ 1.2 ,  0.03,  0.9 ,  0.9 ]),
                         'range_': np.array([ 0.4 ,  0.65]), 'limits': [np.array([-0.255,  4.   ]), np.array([ 0.,  4.]), np.array([ 0.,  1.]), np.array([ 0.,  1.])]}
    param_dict, aero = evaluate_spectra(config,logger)

    ax1 = aero.result.plot_residuals(ax=ax1, datafmt='b')
    ax2.plot(aero.spectra['wave_nm'], aero.spectra['spectra'])
    ax2.plot(aero.param_dict['wave_range'], aero.result.best_fit, 'b-', label='second')
    ax2.errorbar(aero.param_dict['wave_range'], aero.param_dict['spectra_range'], yerr=aero.param_dict['std'], ecolor='g')
    ax2.set_title('Fitted reflectance')
    ax2.set_ylabel('Reflectance')
    ax2.set_xlabel(r'Wavelength $\left[\mu m\right]$')
    ax2.legend()
    plt.show()


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
