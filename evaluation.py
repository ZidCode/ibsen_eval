#!/usr/bin/env python
import re
import logging
import ConfigParser
import numpy as np
from datetime import datetime
from ast import literal_eval
from processing.model_factory import WeatherAtmosphereParameter
from processing.spectrum_analyser import Aerosol_Retrievel
from parser.ibsen_parser import parse_ibsen_file
from utils.plotting import plot_meas, plot_used_irradiance_and_reflectance, plot_fitted_reflectance
from processing.ProcessFactory import DataProcess
import lmfit
"""
    Irradiance measurements does not show calibrated spectra
    Fit class necessary
"""


convert_to_array = lambda x, m : np.array([m(s) for s in x.split(',')])
convert_to_dict = lambda independent: {m.split(':')[0]:float(m.split(':')[1]) for m in independent.split(',')}


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
    config_dict['Fitting']['independent'] = convert_to_dict(config_dict['Fitting']['independent'])
    config_dict['Fitting']['jac_flag'] = literal_eval(config_dict['Fitting']['jac_flag'])
    config_dict['Validation']['validate'] = convert_to_array(config_dict['Validation']['validate'], str)
    config_dict['Validation']['aod_range'] = convert_to_array(config_dict['Validation']['aod_range'], int)
    return config_dict


def create_logger(log_config):
    logger = logging.getLogger('eval')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s - %(funcName)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(log_config['logging_level'])
    return logger


def evaluate_spectra(config, logger=logging):
    ref = parse_ibsen_file(config['Data']['reference'])
    tar = parse_ibsen_file(config['Data']['target'])

    if tar['UTCTime']:
        logger.warning("Config UTCTime: %s. New UTCTime %s from IbsenFile." % (config['Processing']['utc_time'], tar['UTCTime']))
        config['Processing']['utc_time'] = dict()
        config['Processing']['utc_time']['tar'] = tar['UTCTime']
        config['Processing']['utc_time']['ref'] = ref['UTCTime']
    logger.info("Tar Date: %s " % config['Processing']['utc_time']['tar'])
    logger.info("Ref Date: %s " % config['Processing']['utc_time']['ref'])
    logger.info("GPS coords (lat, lon) %s %s" % (config['Processing']['gps_coords'][0], config['Processing']['gps_coords'][1]))
    logger.info("Files\n \t ref: %s  \n \t tar: %s \n" %(config['Data']['reference'], config['Data']['target']))


    Data = DataProcess(config['Fitting']['model'], logger)()
    data_dict = Data.process(ref, tar)
    plot_meas(tar, ref)
    if config['Processing']['logging_level'] == 'DEBUG':
        plot_used_irradiance_and_reflectance(tar, ref, data_dict)
        WeatherParams = WeatherAtmosphereParameter(logger, config, ref['wave'])

    aero = Aerosol_Retrievel(WeatherParams, config['Fitting'], data_dict, logger)
    result, param_dict = aero.getParams()
    logger.info("%s \n" % result.fit_report())
    logger.info("%s \n" % result.success)

    if config['Processing']['logging_level'] == 'DEBUG':
        plot_fitted_reflectance(result, param_dict, data_dict)
    return param_dict, result


def evaluate_measurements(directory, config, logger=logging, output_file='RENAME_ME.csv'):
    import glob
    import pandas as pd
    file_prefixes = ['target', 'reference']
    files = [file_ for file_ in glob.iglob(directory + '%s*' % file_prefixes[0])]
    keys_for_param_dict = ['sun_zenith','utc_times', 'alpha', 'alpha_stderr', 'beta', 'beta_stderr', 'g_dsa', 'g_dsa_stderr', 'g_dsr', 'g_dsr_stderr']
    result_timeline = {key:np.array([]) for key in keys_for_param_dict}

    for file_ in sorted(files):
        for key in file_prefixes:
            config['Data'][key] = file_.replace(file_prefixes[0], key)
        try:
            print(config)
            pass
            logger.info("Evaluating file: %s \n" % file_)
            params, result = evaluate_spectra(config, logger)
            while raw_input('Change initial (y or n)') == 'y':
                for idx, value in enumerate(config['Fitting']['initial_values']):
                    config['Fitting']['initial_values'][idx] = float(raw_input('Old: %s in %s' % (value,idx)))
                params, result = evaluate_spectra(config, logger)

            result_timeline['utc_times'] = np.append(result_timeline['utc_times'], config['Processing']['utc_time']['tar'])
            result_timeline['sun_zenith'] = np.append(result_timeline['sun_zenith'], params['sun_zenith'])
            result_timeline['alpha'] = np.append(result_timeline['alpha'], params['alpha']['value'])
            result_timeline['alpha_stderr'] = np.append(result_timeline['alpha_stderr'], params['alpha']['stderr'])
            result_timeline['beta'] = np.append(result_timeline['beta'], params['beta']['value'])
            result_timeline['beta_stderr'] = np.append(result_timeline['beta_stderr'], params['beta']['stderr'])
            result_timeline['g_dsa'] = np.append(result_timeline['g_dsa'], params['g_dsa']['value'])
            result_timeline['g_dsa_stderr'] = np.append(result_timeline['g_dsa_stderr'], params['g_dsa']['stderr'])
            result_timeline['g_dsr'] = np.append(result_timeline['g_dsr'], params['g_dsr']['value'])
            result_timeline['g_dsr_stderr'] = np.append(result_timeline['g_dsr_stderr'], params['g_dsr']['stderr'])
        except IOError:
            logger.error("%s have no corresponding reference" % file_)

    frame = pd.DataFrame(result_timeline)
    frame.to_csv(output_file, index=False)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='Pass ini-file for processing configurations')
    parser.add_argument('-m', '--measurement_directory', help='Define measurement directory to sweep through')
    parser.add_argument('-o', '--output_file', help='Write timeline results into output file')
    args = parser.parse_args()
    config = parse_ini_config(args.config)
    logger = create_logger(config['Processing'])

    if args.measurement_directory:
        evaluate_measurements(args.measurement_directory, config, logger, args.output_file)
    else:
        evaluate_spectra(config, logger)
