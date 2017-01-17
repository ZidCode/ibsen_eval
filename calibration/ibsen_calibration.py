import os
import copy
import glob
import re
import numpy as np
import pandas as pd
import parser.ibsen_parser as ip
from extract_nonlinearity import generate_nonlinear_correction, check_nonlinearity


def sort_ibsen_by_int(dirname):
    """ marshal, xml, or json pickel packages - todo"""
    """
    Return:
        caldict:
            cal_dict.keys() = [IntTime1, IntTime2, ..]
            cal_dict[IntTime*].keys() = ['darkcurrent', 'reference']
            cal_dict[IntTime*][Type].keys() =  ['num_of_meas', 'data_sample_std', 'data_mean', 'data', 'start_data_index', 'data_std',
                               'wave', 'UTCTime', 'darkcurrent_corrected', 'tdata', 'IntTime', 'Type']
    """
    cal_dict = {}
    for file_ in glob.iglob('%s*.asc' % dirname):
        ibsen_dict = ip.parse_ibsen_file(file_)
        # Skip saturated pixel
        ibsen_dict['wave'] = ibsen_dict['wave'][50:]
        ibsen_dict['mean'] = ibsen_dict['mean'][50:]
        ibsen_dict['tdata'] = ibsen_dict['tdata'][:, 50:]
        print(len(ibsen_dict['wave']))
        file_ = filter(None, file_.split(dirname))[0]
        file_key = filter(None, re.split('[0-9]{3,}\.asc', file_))[0]
        try:
            cal_dict[ibsen_dict['IntTime']][file_key]= ibsen_dict
        except KeyError:
            cal_dict[ibsen_dict['IntTime']] = dict()
            cal_dict[ibsen_dict['IntTime']][file_key] = ibsen_dict
    return cal_dict


def subtract_dark_from_mean(darkcurrent, spectra):
    dd = copy.deepcopy(darkcurrent)
    assert dd['Type'] == 'darkcurrent', 'First parameter has to be darkcurrent'
    dd['mean'] = ip.get_mean_column(dd)
    if spectra['darkcurrent_corrected'] == False:
        spectra['mean'] = ip.get_mean_column(spectra)
        spectra['tdata'] -= dd['mean']
        spectra['data'] = np.transpose(spectra['tdata'])
        spectra['mean'] -= dd['mean']
        spectra['darkcurrent_corrected'] = True


def subtract_dark(cal_dict):
    for key, item in cal_dict.items():
        subtract_dark_from_mean(item['darkcurrent'], item['reference'])
    return cal_dict


def read_file(offset_file, cal_dict=None):
    noise_dict = dict()
    print("Bias file detected in %s" % offset_file)
    data = np.genfromtxt(offset_file, skip_header=1, delimiter=',')
    noise_dict['channel'] = data[:, 0]
    noise_dict['noise'] = data[:, 1]
    return noise_dict


def calc_offset(offset_file, cal_dict):
    noise_dict = dict()
    sorted_keys = sorted(cal_dict.keys())
    tmp_channels = range(len(cal_dict[sorted_keys[0]]['darkcurrent']['wave']))
    IntTimes = np.array(sorted_keys)
    noise = np.array([])
    for channel in tmp_channels:
        noise_dict[channel] = dict()
        dark = np.array([cal_dict[key]['darkcurrent']['mean'][channel] for key in sorted_keys])
        noise_dict[channel]['dark'] = dark
        coeffs_dark = np.polyfit(sorted_keys, dark, deg=1)
        noise = np.append(noise, coeffs_dark[1])
    noise_dict['noise'] = noise
    noise_dict['channel'] = tmp_channels
    frame = pd.DataFrame(np.transpose([tmp_channels, noise]), columns=['ch', 'bias'])
    frame.to_csv(offset_file, index=False)
    return noise_dict


def get_noise(flag):
    if flag:
        return read_file
    else:
        return calc_offset


def generate_ibsen_calibration_files(directory, reference):
    # Extract Rasta specific raw data
    cal_dict = sort_ibsen_by_int(directory)
    cal_dict_tmp = copy.deepcopy(cal_dict)
    bias_file = directory + 'assumed_bias/' + 'bias.txt'
    flag = os.path.exists(bias_file) 
    noise_dict = get_noise(flag)(bias_file, cal_dict)

    # Substract darkcurrent from measurements
    nonlinear_config = {'max_lowest_int_time': 2323, 'sigma': 10, 'index_start_spline_fit': 700, 'gaussian_mean_steps':4}
    nonlinear_correction_dict = generate_nonlinear_correction(cal_dict_tmp, nonlinear_config, noise_dict)
    noise_dict['noise'] = np.zeros(len(noise_dict['channel']))  # Setting offset to zero
    cal_dict_tmp = subtract_dark(cal_dict_tmp)
    nonlinear_correction_dark = generate_nonlinear_correction(cal_dict_tmp, nonlinear_config, noise_dict)
    check_nonlinearity(cal_dict, [nonlinear_correction_dict, nonlinear_correction_dark])

    while raw_input('Change settings (y or n)') == 'y':
        for key in nonlinear_config.keys():
            nonlinear_config[key] = int(raw_input('%s' %key))
        nonlinear_correctiony_dict = generate_nonlinear_correction(cal_dict, nonlinear_config)
        check_nonlinearity(cal_dict, nonlinear_correction_dict)

    #Nonlinear correction for ibsen response
    for integration, spectra in cal_dict.items():
        spectra['reference']['mean'] = spectra['reference']['mean'] / np.interp(spectra['reference']['mean'], nonlinear_correction_dict['DN'], nonlinear_correction_dict['nonlinear'])
        spectra['reference']['mean'] = spectra['reference']['mean'] / integration
    # Generate ibsen response factors for physical units
    cal_dict, response_dict = generate_response_factors(cal_dict, reference)
    import matplotlib.pyplot as plt
    for integration, spectra in cal_dict.items():
        spectra['reference']['mean'] = spectra['reference']['mean'] / np.interp(spectra['reference']['wave'], response_dict['wave'], response_dict['scale_factors'])
        plt.plot(spectra['reference']['wave'], spectra['reference']['mean'])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel(r'$\frac{mW}{nm m^2 sr}$')
    plt.show()
    
    plt.plot(response_dict['wave'], response_dict['intensity'], 'r+', label='Ibsen response')
    plt.plot(response_dict['wave'], response_dict['halogen'], 'b+', label='Halogen lamp')
    plt.plot(response_dict['wave'], response_dict['intensity'] / response_dict['scale_factors'], 'y', label='Calibrated ibsen response')
    plt.xlabel('Wavelength [nm]')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """Usage:
        python ibsen_calibration.py -d /home/joanna/DLR/Codes/calibration/Ibsen_0109_5313264/EOC/Optiklabor/
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='/home/joanna/DLR/Codes/calibration/Ibsen_0109_Serialnumber_missing/EOC/Optiklabor/', help="Add directory with raw data measured by Rasta")
    parser.add_argument('-r', '--reference_file', default='/home/joanna/DLR/Codes/calibration/GS1032_1m.txt',help="Reference file for halogen lamp")
    args = parser.parse_args()
    print(args.reference_file)
    generate_ibsen_calibration_files(args.directory, args.reference_file)
