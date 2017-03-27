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
from matplotlib.font_manager import FontProperties


def darkcurrent_channel_analyse(directory):
    hfont = {'family':'serif', 'fontsize': 12}

    cal_dict = sort_ibsen_by_int(directory)
    for key, val in sorted(cal_dict.items()):
        plt.plot(val['darkcurrent']['wave'], val['darkcurrent']['mean'], label='%1.f' % key)
    plt.xlabel(r'Wavelength $\lambda$ [nm]', **hfont)
    plt.ylabel(r'Signal [DN]', **hfont)

    fontP = FontProperties()
    fontP.set_family('serif')
    fontP.set_size('small')
    legend = plt.legend(loc=0, ncol=1, prop = fontP,fancybox=True,
                        shadow=False,title='Integration Times [ms]',bbox_to_anchor=(1.0, 1.0))
    #plt.setp(legend.get_title(),fontsize='small')
    plt.tight_layout()
    plt.show()

    plt.plot(cal_dict[5]['darkcurrent']['wave'], cal_dict[5]['reference']['mean'], label='Mean')
    plt.plot(cal_dict[5]['darkcurrent']['wave'], cal_dict[5]['darkcurrent']['mean'], label='Mean')
    plt.plot(cal_dict[5]['darkcurrent']['wave'], cal_dict[5]['darkcurrent']['data'], alpha=0.05)
    plt.plot(cal_dict[5]['darkcurrent']['wave'], cal_dict[5]['reference']['data'], alpha=0.05)
    plt.xlabel(r'Wavelength $\lambda$ [nm]', **hfont)
    plt.ylabel(r'Signal [DN]', **hfont)
    plt.show()

    sorted_keys = sorted(cal_dict.keys()) #[4:-4]
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

    #ax1.plot(IntTimes, noise_dict[258]['dark'])
    ax2.plot(tmp_channels, noise)
    ax1.set_xlabel('Integration Time [ms]', **hfont)
    ax2.set_xlabel(r'Wavelength $\lambda$ [nm]', **hfont)
    ax1.set_ylabel('Signal [DN]', **hfont)
    ax2.set_ylabel('Signal [DN]', **hfont)
    plt.tight_layout()
    plt.show()

def temperature():
    FONTSTYLE = 'serif'
    FONTSIZE = 12
    hfont = {'family':FONTSTYLE, 'fontsize': FONTSIZE}
    fontP = FontProperties()
    fontP.set_family(FONTSTYLE)
    fontP.set_size('small')

    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0, :])

    winter = '/home/joanna/DLR/Codes/measurements/measurements/LMU/291116_LMU/'
    summer = '/home/joanna/DLR/Codes/measurements/measurements/Roof_DLR/2016_09_14RoofDLR/'
    import glob
    file_prefixes = ['darkcurrent']
    files_winter = sorted([file_ for file_ in glob.iglob(winter + '%s*' % file_prefixes[0])])[5:7]
    files_summer = sorted([file_ for file_ in glob.iglob(summer + '%s*' % file_prefixes[0])])[3:5]
    files_winter = [files_winter[0]]
    for win in files_winter:
        print(win)
        win_dict = ip.parse_ibsen_file(win)
        ax1.plot(win_dict['wave'][50:], win_dict['mean'][50:],color='darkblue', label='-3 $^{\circ}$C' % win_dict['IntTime'])

    files_summer = [files_summer[0]]
    for sum in files_summer:
        print(sum)
        sum_dict = ip.parse_ibsen_file(sum)
        ax1.plot(sum_dict['wave'][50:], sum_dict['mean'][50:],color='sandybrown', label='30 $^{\circ}$C' % sum_dict['IntTime'])
    assert win_dict['IntTime'] == sum_dict['IntTime']
    ax1.set_ylabel('Signal [DN]', **hfont)
    ax1.set_ylabel('Signal [DN]', **hfont)
    ax1.set_xlabel(r'Wavelength $\lambda$ [nm]', **hfont)
    ax1.legend(loc='best', prop=fontP, title=r' %s ms Integrationtime at temperature:' % win_dict['IntTime'])
    plt.setp(ax1.get_legend().get_title(), fontsize='small', family=FONTSTYLE)
    plt.tight_layout()
    plt.show()


def test_offset_subtraction():
    """ testing of offset subtraction or darkcurrent subtraction is better"""
    test_directory = '/home/joanna/DLR/Codes/calibration/test_nonlinearity/labor/ibsen_nonlinearity_verschoeben2/'
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
    parser.add_argument('-d', '--directory', default='/home/joanna/DLR/Codes/calibration/Ibsen_0109_5313264/EOC/Optiklabor/', help="Add directory with raw data measured by Rasta")
    args = parser.parse_args()
    test_offset_subtraction()
    darkcurrent_channel_analyse(args.directory)
    temperature()
