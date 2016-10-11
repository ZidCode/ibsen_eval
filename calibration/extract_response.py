import numpy as np
import pandas as pd


def get_halogen_spectra(reference_file):
    d = np.genfromtxt(reference_file, delimiter=',')
    wavelength = d[:, 0]
    intensity_mW = d[:, 1] * 10 ** -6  #[W/(m3sr) into mW/(nmm2sr)]
    relative_error = d[:, 2]
    return wavelength, intensity_mW, relative_error


def calc_scaling_factors(waveibs, ref, reference_file):
    wave, intensity, r_err = get_halogen_spectra(reference_file)

    start_ind = np.where(waveibs > wave[0])[0][0]
    end_ind = np.where(waveibs < wave[-1])[0][-1]

    mod_waves = waveibs[start_ind:end_ind]
    mod_intensity = ref[start_ind:end_ind]
    assert len(mod_waves) == len(mod_intensity)
    assert ref[start_ind] == mod_intensity[0]

    map_holgen_intensities = np.interp(mod_waves, wave, intensity)
    scale_factor = mod_intensity / map_holgen_intensities
    return scale_factor, mod_intensity, map_holgen_intensities, mod_waves


def generate_response_factors(cal_dict, halogen_file):
    store_to_file = 'response.txt'
    spectras = np.array([])
    for integration, spectra in cal_dict.items():
        assert spectra['reference']['darkcurrent_corrected'] == True
        try:
            spectras = np.vstack((spectras, spectra['reference']['mean']))
        except ValueError:
            wave = spectra['reference']['wave']
            spectras = spectra['reference']['mean']

    mean_spectra = np.mean(spectras, axis=0)
    scale_factor, mod_intensity, map_holgen_intensities, mod_waves = calc_scaling_factors(wave, mean_spectra, halogen_file)
    response_dict = {'scale_factors': scale_factor, 'wave': mod_waves, 'halogen':map_holgen_intensities, 'intensity': mod_intensity}

    frame = pd.DataFrame(np.transpose([mod_waves, scale_factor, mod_intensity, map_holgen_intensities]),
                         columns=['Wavelength', 'ScaleFactor', 'Intensity', 'HalogenIntensity'])
    frame.to_csv(store_to_file, index=False)
    import matplotlib.pyplot as plt
    plt.plot(mod_waves, scale_factor)
    plt.ylabel(r'$\frac{DN}{\frac{mW}{nm m^2 sr}}$')
    plt.xlabel('Wavelength [nm]')
    plt.show()

    return cal_dict, response_dict

