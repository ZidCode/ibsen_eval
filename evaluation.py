#!/usr/bin/env python
import os
import copy
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from processing.spectrum_analyser import get_spectral_irradiance_reflectance
from processing.solar_zenith import get_sun_zenith
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


# Decorators
def evaluate(rfile, tfile, dfile, gps_coords, utc_time,  check=False):

    ref = parse_ibsen_file(rfile)
    tar = parse_ibsen_file(tfile)
    dark = parse_ibsen_file(dfile)

    ref_data = copy.deepcopy(ref)
    tar_data = copy.deepcopy(tar)
    subtract_dark_from_mean(dark, tar, ref)

    assert tar['mean'].all() == (get_mean_column(tar) - get_mean_column(dark)).all()
    assert ref['data'][:, 0].all() == ref['tdata'][0].all()

    reflectance = get_spectral_irradiance_reflectance(ref['mean'], tar['mean'])
    sun_zenith = get_sun_zenith(utc_time, *gps_coords)

    print("Files\n \t ref: %s  \n \t tar: %s \n \t dark: %s" %(rfile, tfile, dfile))
    print("Date: %s \n \t Zenith angle %s" %(utc_time, sun_zenith))

    if check:
        assert not tar_data['data'].all() == tar['data'].all()
        plot_meas(tar_data, ref_data, dark)
        frame = pd.DataFrame(np.transpose([tar['wave'], reflectance]), columns=['Wavelength', 'Reflectance'])
        frame.to_csv('reflectance.csv', index=False)
        plot_used_irradiance_and_reflectance(tar, ref, reflectance)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help="Set debug level")
    args = parser.parse_args()

    #minimaleCirren T4/ST06/target000.asc
    measurement = os.path.dirname(os.path.realpath(__file__)) + '/../measurements/Ostsee/T4/ST06/'

    gps_coords = [53.9453236, 11.3829424, 0]
    utc_time = datetime.strptime('2016-04-14 08:47:00', '%Y-%m-%d %H:%M:%S')
    files = ['reference000.asc', 'target000.asc','darkcurrent000.asc']
    file_set  = np.array([measurement + f for f in files])
    file_set = np.append(file_set, [gps_coords, utc_time, args.debug])
    evaluate(*file_set)
