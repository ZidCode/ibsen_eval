import os
import glob
import re
import numpy as np
import pandas as pd
from string import replace
import ibsen_parser as ip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_halogen_spectra(filename=os.path.dirname(os.path.realpath(__file__))+'/../../calibration/GS1032_1m.txt'):
    d = np.genfromtxt(filename, delimiter=',')
    wavelength = d[:, 0]
    intensity = d[:, 1]
    relative_error = d[:, 2]
    return wavelength, intensity, relative_error


def sort_ibsen_by_int(dirname=os.path.dirname(os.path.realpath(__file__))+"/../../calibration/Rasta/"):
    """ marshal, xml, or json pickel packages - todo"""
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


def calc_scaling_factors(waveibs, dark, ref):
    wave, intensity, r_err = get_halogen_spectra()

    start_ind = np.where(waveibs > wave[0])[0][0]
    end_ind = np.where(waveibs < wave[-1])[0][-1]

    mod_waves = waveibs[start_ind:end_ind]
    mod_dark = dark[start_ind:end_ind]
    mod_ref = ref[start_ind:end_ind]
    assert len(mod_waves) == len(mod_dark) == len(mod_ref)
    assert ref[start_ind] == mod_ref[0]

    mod_intensity = mod_ref - mod_dark
    map_holgen_intensities = np.interp(mod_waves, wave, intensity)
    scale_factor = map_holgen_intensities / mod_intensity * 10 ** -6
    return scale_factor, mod_intensity, map_holgen_intensities, mod_waves


def generate_ibsen_cal():
    # Generates calibration files with Scale, Ref and Dark (all mean)

    cal_dict = sort_ibsen_by_int()
    for key, item in cal_dict.items():
        #TODO Get rid off directory
        store_to_file = '../Calibration_Values/ScaleCal_IntTime_%s' % int(key)
        assert item['darkcurrent']['wave'].all() == item['reference']['wave'].all()
        wave = item['reference']['wave']

        item['darkcurrent']['mean'] = ip.get_mean_column(item['darkcurrent'])
        item['reference']['mean'] = ip.get_mean_column(item['reference'])
        sf, mod_int, map_hal, w = calc_scaling_factors(wave, item['darkcurrent']['mean'], item['reference']['mean'])
        frame = pd.DataFrame(np.transpose([w, sf, mod_int, map_hal]),
                             columns=['Wavelength', 'ScaleFactor', 'Intensity', 'HalogenIntensity'])
        frame.to_csv(store_to_file, index=False)
    return cal_dict


def check_nonlinearity(min=8, max=16, step=2):
    cal_dict = generate_ibsen_cal()
    wave, int, r_err = get_halogen_spectra()
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    sorted_keys = sorted(cal_dict.keys())
    chosen_keys = [key for key in sorted_keys[min:max:step]]
    for key in chosen_keys:
        ax1.plot(cal_dict[key]['reference']['wave'], (cal_dict[key]['reference']['mean'] - cal_dict[key]['darkcurrent']['mean']) / key, label='%s' %key)
        ax2.plot(cal_dict[key]['darkcurrent']['wave'], cal_dict[key]['darkcurrent']['mean'] / key, label='%s' %key)
    ax1.set_title('Response')
    ax2.set_title('Darkcurrent')
    ax1.set_ylabel('DN')
    ax2.set_ylabel('DN')
    ax2.set_xlabel('Wavelength [nm]')
    ax2.legend(title='IntTime [ms]', bbox_to_anchor=(1.15, 1.6))
    plt.show()


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', help="Add calibration file")
    parser.add_argument('-c', '--check', action="store_true", help="Check calibration, Plots will be shown")
    parser.add_argument('-r','--range', nargs='+', default=['1', '-1', '1'],  help='min, max, step to determine calibration files (-c has to added)')
    args = parser.parse_args()
    conv_to_float = lambda x: int(x)
    # TODO independancy of integration time
    if args.check:
        range = map(conv_to_float, args.range)
        check_nonlinearity(*range)
    else:
        generate_ibsen_cal()
