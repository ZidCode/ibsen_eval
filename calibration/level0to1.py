import numpy as np
import ibsen_calibration as ic
from parser.ibsen_parser import parse_ibsen_file
"This module will be deleted"

def calibrate_meas(data_file, dark_file, nonlinear_correction_file, response_file):
    import matplotlib.pyplot as plt
    non_linear = np.genfromtxt(nonlinear_correction_file, skip_header=1, delimiter=',')
    DN = non_linear[:, 0]
    correction_values = non_linear[:, 1]

    response = np.genfromtxt(response_file, skip_header=1, delimiter=',')
    wave = response[:,0]
    scale_factors = response[:,1]

    data_dict = parse_ibsen_file(data_file)
    dark_dict = parse_ibsen_file(dark_file)
    ic.subtract_dark_from_mean(dark_dict, data_dict)
    assert data_dict['darkcurrent_corrected'] == True
    data_dict['tdata'] = data_dict['tdata'] / np.interp(data_dict['tdata'], DN, correction_values)
    data_dict['data'] = np.transpose(data_dict['tdata'])
    data_dict['tdata'] = data_dict['tdata'] / data_dict['IntTime']
    data_dict['mean'] = data_dict['mean'] / np.interp(data_dict['mean'], DN, correction_values)
    data_dict['mean'] = data_dict['mean'] / data_dict['IntTime']
    data_dict['data'] = np.transpose(data_dict['tdata'])
    data_dict['tdata']= np.divide(data_dict['tdata'] , np.interp(data_dict['wave'], wave, scale_factors))
    data_dict['data'] = np.transpose(data_dict['tdata'])
    return data_dict


def process_level0to1(meas_files, dark_file, nonlinear_correction_file, response_file):

    for meas_file in meas_files:
        print('File to calibrate %s \n' % meas_file.split('/')[-1])
        print('\t with Dark %s \n' % dark_file.split('/')[-1])
        cal_dict = calibrate_meas(meas_file, dark_file, nonlinear_correction_file, response_file)
        write_to_file(cal_dict, meas_file)


def write_to_file(cal_dict, filename):

    assert cal_dict['darkcurrent_corrected'] == True
    file_list = np.insert(filename.split('/'), -1, 'calibrated')
    file_ = '/'.join(file_list)
    print(file_)
    data = np.transpose(np.vstack((cal_dict['wave'], cal_dict['data_mean'], cal_dict['data_std'],cal_dict['tdata'])))
    with open(filename, 'r') as fr, open(file_, 'w') as fp:
        old = fr.readlines()
        old[cal_dict['start_data_index'] - 1 ] = '[DataCalibrated]\n'
        fp.writelines(old[:cal_dict['start_data_index']])
        for dat in data:
            dat.tofile(fp, sep='\t', format="%.4f")
            fp.write('\n')


def start_level0to1(directory, nonlinear_correction_file, response_file):
    import glob
    file_prefixes = ['darkcurrent', 'reference', 'target']
    files = [file_ for file_ in glob.iglob(directory + '%s*' % file_prefixes[0])]

    for file_ in sorted(files):
        index = file_.split(file_prefixes[0])[-1]
        files = [f for f in glob.iglob(directory + '*%s' % index) if file_prefixes[0] not in f]
        process_level0to1(files, file_, nonlinear_correction_file, response_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='/home/jana_jo/DLR/Codes/measurements/Roof_DLR/2016_09_13RoofDLR/test/', help='Define measurement directory to sweep through')
    parser.add_argument('-n', '--nonlinear',
                        default='/home/jana_jo/DLR/Codes/evaluation/calibration/Ibsen_0109_5313264_calibration_files/nonlinearity_correction.txt',
                        help='Nonlinear correction file for corresponding ibsen')
    parser.add_argument('-r', '--response', default='/home/jana_jo/DLR/Codes/evaluation/calibration/Ibsen_0109_5313264_calibration_files/response.txt',
                        help='Response file for corresponding ibsen')
    args = parser.parse_args()
    start_level0to1(args.directory, args.nonlinear, args.response)
