import copy
import glob
import re
import numpy as np
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
        spectra['mean'] -= dd['mean']
        spectra['darkcurrent_corrected'] = True


def subtract_dark(cal_dict):
    for key, item in cal_dict.items():
        subtract_dark_from_mean(item['darkcurrent'], item['reference'])
    return cal_dict


def generate_ibsen_calibration_files(directory):
    # Extract Rasta specific raw data
    cal_dict = sort_ibsen_by_int(directory)
    # Substract darkcurrent from measurements
    cal_dict = subtract_dark(cal_dict)
    nonlinear_correction_dict = generate_nonlinear_correction(cal_dict)
    check_nonlinearity(cal_dict, nonlinear_correction_dict)
    # Nonlinear correction for ibsen response
    for integration, spectra in cal_dict.items():
        spectra['reference']['mean'] = spectra['reference']['mean'] / np.interp(spectra['reference']['mean'], nonlinear_correction_dict['DN'], nonlinear_correction_dict['nonlinear'])
        spectra['reference']['mean'] = spectra['reference']['mean'] / integration
    # Generate ibsen response factors for physical units


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help="Add directory with raw data measured by Rasta")
    args = parser.parse_args()
    print(args.directory)
    generate_ibsen_calibration_files(args.directory)
