#!/usr/bin/env python
import os
import copy
import logging
import ConfigParser
import numpy as np
import pandas as pd
from datetime import datetime
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from processing.spectrum_analyser import get_spectral_irradiance_reflectance
import processing.irradiance_models as irr
from parser.ibsen_parser import parse_ibsen_file, get_mean_column, get_mean_column, subtract_dark_from_mean
"""
FREEDOM VIS - Ibsen
    360 to 830 nm wavelength range
    Numerical aperture of 0.16
    Minimum resolution of 1.3 nm (FWHM)
    Footprint of 25 mm x 48 mm
ibsen_dict:
    {'num_of_meas': <int>,
    'data_mean': array([..]),
    'tdata': array([[..],..,[..]]) shape(30, 1024),
    'start_data_index': 18,
    'data_std': array([..]),
    'wave': array([..]),
    'IntTime': <float>,
    'data': array([[..],..,[..]]) shape(1024, 30)}
"""


def plot_meas(tar, ref, dark):
    gs = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    ax3 = plt.subplot(gs[2, :])
    map_dicts = {'Target_E_ds': {'plt': ax1, 'meas': tar},
                 'Reference_E_dd': {'plt': ax2, 'meas': ref},
                 'Darkcurrent': {'plt': ax3, 'meas':dark}}

    for k, v in map_dicts.items():
        v['plt'].plot(v['meas']['wave'], v['meas']['data'])
        v['plt'].set_title('%s' % k)

    plt.tight_layout()
    plt.xlabel('Wavelength [nm]')
    plt.show()


def plot_used_irradiance_and_reflectance(tarmd, refmd, reflectance):
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    map_dict = {'E_ds_mean': tarmd, 'E_dd_mean': refmd}

    for key, meas in map_dict.items():
        ax1.plot(meas['wave'], meas['data'], alpha=0.1)
        ax1.plot(meas['wave'], meas['mean'], label='%s' %key)

    ax1.set_title('Diffuse and direct Irradiance')
    ax2.plot(tarmd['wave'], reflectance)
    ax2.set_title('Reflectance')
    ax1.legend()
    plt.xlabel('Wavelength [nm]')
    plt.show()


def parse_ini_config(ini_file):
    config = ConfigParser.ConfigParser()
    config.read(ini_file)
    config_dict = {s: dict(config.items(s)) for s in config.sections()}
    config_dict['Processing']['logging'] = literal_eval(config_dict['Processing']['logging'])
    config_dict['Data']['gps_coords'] = np.array([float(s) for s in config_dict['Data']['gps_coords'].split(',')])
    config_dict['Data']['utc_time'] = datetime.strptime(config_dict['Data']['utc_time'], '%Y-%m-%d %H:%M:%S')
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
    config_data = config['Data']
    ref = parse_ibsen_file(config_data['reference'])
    tar = parse_ibsen_file(config_data['target'])
    dark = parse_ibsen_file(config_data['dark'])

    ref_data = copy.deepcopy(ref)
    tar_data = copy.deepcopy(tar)
    subtract_dark_from_mean(dark, tar, ref)

    assert tar['mean'].all() == (get_mean_column(tar) - get_mean_column(dark)).all()
    assert ref['data'][:, 0].all() == ref['tdata'][0].all()

    reflectance = get_spectral_irradiance_reflectance(ref['mean'], tar['mean'])
    irradiance_model = irr.build_Model(config['Data'], logger)
    reflectance_dict = {'wave_mu': ref['wave'] / 1000.0, 'reflect': reflectance}

    logger.info("Files\n \t ref: %s  \n \t tar: %s \n \t dark: %s" %(config_data['reference'], config_data['target'], config_data['dark'] ))
    logger.info("Date: %s \n \t Zenith angle %s" %(config_data['utc_time'], sun_zenith))

    if config['Processing']['logging_level'] == 'DEBUG':
        assert not tar_data['data'].all() == tar['data'].all()
        plot_meas(tar_data, ref_data, dark)
        frame = pd.DataFrame(np.transpose([tar['wave'], reflectance]), columns=['Wavelength', 'Reflectance'])
        frame.to_csv('reflectance.csv', index=False)
        plot_used_irradiance_and_reflectance(tar, ref, reflectance)


if __name__ == "__main__":
    default_ini = 'config.ini'
    config = parse_ini_config(default_ini)
    evaluate(config)
