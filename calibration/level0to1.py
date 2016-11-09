import numpy as np
import ibsen_calibration as ic
from parser.ibsen_parser import parse_ibsen_file
"This module will be deleted"


def process_level0to1(meas, dark):

    for meas_file in meas:
        print(meas_file)

    #parse files
    #subtract dark from meas
    # divide by nonlinearity
    # divide by responst
    # change values

def write_to_file(cal_dict, filename):

    assert cal_dict['darkcurrent_corrected'] == True
    file_ = 'calibrated/' + filename
    data = np.transpose(np.vstack((cal_dict['wave'], cal_dict['data_mean'], cal_dict['data_std'],cal_dict['tdata'])))
    with open(filename, 'r') as fr, open(file_, 'w') as fp:
        old = fr.readlines()
        old[cal_dict['start_data_index'] - 1 ] = '[DataCalibrated]\n'
        fp.writelines(old[:cal_dict['start_data_index']])
        for dat in cal_dict['data']:
            dat.tofile(fp, sep='\t',format="%.4f")
            fp.write('\n')



def start_level0to1(directory):
    import glob
    file_prefixes = ['darkcurrent', 'reference', 'target']
    files = [file_ for file_ in glob.iglob(directory + '%s*' % file_prefixes[0])]

    for file_ in sorted(files):
        index = file_.split(file_prefixes[0])[-1]
        files = [f for f in glob.iglob(directory + '*%s' % index) if file_prefixes[0] not in f]
        process_level0to1(files, file_)
        #print("files %s and corr. dark file %s \n" % (files, file_))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='/home/jana_jo/DLR/Codes/measurements/Roof_DLR/2016_09_13RoofDLR/', help='Define measurement directory to sweep through')
    args = parser.parse_args()
    start_level0to1(args.directory)
