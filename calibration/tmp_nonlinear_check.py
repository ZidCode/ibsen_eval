import copy
import glob
import re
import numpy as np
import parser.ibsen_parser as ip
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from extract_nonlinearity import generate_nonlinear_correction, check_nonlinearity
from extract_response import generate_response_factors
from level0to1 import read_nonlinear_correction_file
from ibsen_calibration import sort_ibsen_by_int



def darkcurrent_channel_analyse(directory):

    cal_dict = sort_ibsen_by_int(directory)
    sorted_keys = sorted(cal_dict.keys())[4:-4]
    tmp_channels = range(len(cal_dict[sorted_keys[0]]['darkcurrent']['wave']))
    noise_dict = dict()
    #tmp_channels = np.delete(tmp_channels, [187, 258, 265, 811])794
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    IntTimes = np.array(sorted_keys)
    noise = np.array([])

    dark_tmp = cal_dict[110]['darkcurrent']['mean']
    #ax1.plot( dark_tmp, '+')
    print(np.where(dark_tmp > 2361 ))

    for channel in tmp_channels:
        noise_dict[channel] = dict()
        dark = np.array([cal_dict[key]['darkcurrent']['mean'][channel] for key in sorted_keys])
        noise_dict[channel]['dark'] = dark
        coeffs_dark = np.polyfit(sorted_keys, dark, deg=1)

        noise = np.append(noise, coeffs_dark[1])
        ax1.plot(IntTimes, dark, '+')



    ax1.plot(IntTimes, noise_dict[258]['dark'])
    ax2.plot(tmp_channels, noise)
    #ax2.plot(cal_dict[110]['darkcurrent']['mean'])
    #ax2.plot(np.ones(2400)*258, np.linspace(0, 2400, 2400))
    ax1.set_title('First Channel')
    ax1.set_xlabel('Integration Time [ms]')
    ax1.set_ylabel('DN [a.u.]')
    ax1.legend(loc='best')
    plt.show()

def temperature():
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0, :])
    #ax2 = plt.subplot(gs[1, :])

    winter = '/home/joanna/DLR/Codes/measurements/measurements/LMU/291116_LMU/'
    summer = '/home/joanna/DLR/Codes/measurements/measurements/Roof_DLR/2016_09_14RoofDLR/'
    import glob
    file_prefixes = ['darkcurrent']
    files_winter = sorted([file_ for file_ in glob.iglob(winter + '%s*' % file_prefixes[0])])[5:7]
    files_summer = sorted([file_ for file_ in glob.iglob(summer + '%s*' % file_prefixes[0])])[3:5]
    for win in files_winter:
        print(win)
        win_dict = ip.parse_ibsen_file(win)
        ax1.plot(win_dict['wave'], win_dict['mean'], label='-3 Grad %s IntTime' % win_dict['IntTime'])

    for sum in files_summer:
        print(sum)
        sum_dict = ip.parse_ibsen_file(sum)
        ax1.plot(sum_dict['wave'], sum_dict['mean'], label='30 Grad %s IntTime' % sum_dict['IntTime'])

    ax1.set_title('Thermisches Rauschen')
    #ax2.set_title('30 Grad Temperatur')
    ax1.set_ylabel('DN')
    ax1.set_ylabel('DN')
    ax1.set_xlabel('Wavelength [nm]')
    ax1.legend(loc='best')
    #ax2.legend(loc='best')
    plt.show()

def test_offset_subtraction():
    """ testing of offset subtraction or darkcurrent subtraction is better"""
    test_directory = '/home/jana_jo/DLR/Codes/calibration/test_nonlinearity/labor/ibsen_nonlinearity_verschoeben2/'
    offset = 'offset_corrected_calibration/'
    dark_ = 'darkcurrent_corrected_calibration/'
    _file = 'nonlinearity_correction.txt'
    nonlinear_correction_dict = dict()
    nonlinear_correction_dict_dark = dict()
    cal_dict = sort_ibsen_by_int(test_directory)
    nonlinear_correction_dict['DN'], nonlinear_correction_dict['nonlinear'] = read_nonlinear_correction_file(offset + _file)
    nonlinear_correction_dict_dark['DN'], nonlinear_correction_dict_dark['nonlinear'] = read_nonlinear_correction_file(dark_ + _file)
    check_nonlinearity(cal_dict, [nonlinear_correction_dict,nonlinear_correction_dict_dark])


if __name__  == "__main__":
    #/home/jana_jo/DLR/Codes/measurements/LMU/291116_LMU               Winter
    #/home/jana_jo/DLR/Codes/measurements/Roof_DLR/2016_08_25_RoofDLR  Sommer

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='/home/jana_jo/DLR/Codes/calibration/Ibsen_0109_Serialnumber_missing/EOC/Optiklabor/', help="Add directory with raw data measured by Rasta")
    args = parser.parse_args()
    test_offset_subtraction()
    #darkcurrent_channel_analyse(args.directory)
    temperature()
