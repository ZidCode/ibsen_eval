import os
import numpy as np


def get_halogen_spectra(filename=os.path.dirname(os.path.realpath(__file__))+'/../../calibration/GS1032_1m.txt'):
    d = np.genfromtxt(filename, delimiter=',')
    wavelength = d[:, 0]
    intensity = d[:, 1]
    relative_error = d[:, 2]
    return wavelength, intensity, relative_error


def calc_scaling_factors(waveibs, ref):
    wave, intensity, r_err = get_halogen_spectra()

    start_ind = np.where(waveibs > wave[0])[0][0]
    end_ind = np.where(waveibs < wave[-1])[0][-1]

    mod_waves = waveibs[start_ind:end_ind]
    mod_ref = ref[start_ind:end_ind]
    assert len(mod_waves) == len(mod_ref)
    assert ref[start_ind] == mod_ref[0]

    mod_intensity = mod_ref
    map_holgen_intensities = np.interp(mod_waves, wave, intensity)
    scale_factor = map_holgen_intensities / mod_intensity  * 10 ** -6
    return scale_factor, mod_intensity, map_holgen_intensities, mod_waves


def generate_ibsen_cal(cal_dict):
    # Generates calibration files with Scale, Ref and Dark (all mean)
    store_to_file = '../Calibration_Values/response.txt'
    response_dict = dict()

    for key, item in cal_dict.items():
        #TODO Get rid off directory
        response_dict[key] = dict()
        assert item['darkcurrent']['wave'].all() == item['reference']['wave'].all()
        wave = item['reference']['wave']

        assert item['reference']['darkcurrent_corrected'] == True

        sf, mod_int, map_hal, w = calc_scaling_factors(wave, item['reference']['mean'] / key)
        response_dict[key]['wave'] = w
        response_dict[key]['ScaleFactor'] = sf

        #frame = pd.DataFrame(np.transpose([w, sf, mod_int, map_hal]),
                             #columns=['Wavelength', 'ScaleFactor', 'Intensity', 'HalogenIntensity'])
        #frame.to_csv(store_to_file, index=False)

    return cal_dict, response_dict

