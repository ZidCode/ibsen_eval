import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import scipy.interpolate as inter
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal


def generate_nonlinear_correction(cal_dict, nonlinear_config, noise_dict):
    nonlinear_correction_dict = dict()
    DN, nonlinear_correction = calculate_nonlinearity_factors(cal_dict, nonlinear_config, noise_dict)
    nonlinear_correction_dict['DN'] = DN
    nonlinear_correction_dict['nonlinear']= nonlinear_correction
    frame = pd.DataFrame(np.transpose([DN, nonlinear_correction]), columns=['DN', 'nonlinear_correction'])
    frame.to_csv('nonlinearity_correction.txt', index=False)
    return nonlinear_correction_dict


def calculate_nonlinearity_factors(cal_dict, nonlinear_config, noise_dict):
    max_lowest_int_time = nonlinear_config['max_lowest_int_time'] # pick value manually. Needs to be slightly above the hightest value of the lowest integration time, WTF?
    sigma = nonlinear_config['sigma']
    index_start_spline_fit = nonlinear_config['index_start_spline_fit']
    gaussian_mean_steps = nonlinear_config['gaussian_mean_steps']

    intTimes_sorted = sorted(cal_dict.keys())
    dn_Matrix_intTimeperrow = cal_dict[intTimes_sorted[0]]['reference']['mean'] - noise_dict['noise']

    for intTime in intTimes_sorted[1:]:
        noise_corrected = cal_dict[intTime]['reference']['mean'] - noise_dict['noise']
        dn_Matrix_intTimeperrow = np.vstack((dn_Matrix_intTimeperrow, noise_corrected))
    DN_Matrix_intTimepercolumn = np.transpose(dn_Matrix_intTimeperrow)
    assert_array_equal(DN_Matrix_intTimepercolumn[:,4], cal_dict[intTimes_sorted[4]]['reference']['mean'] - noise_dict['noise'])

    values = np.array([max(x) for x in DN_Matrix_intTimepercolumn])
    index = np.where(values > max_lowest_int_time)[0]
    data = DN_Matrix_intTimepercolumn[index[0]:max(index)]
    DN_nonlin = np.array([])
    all_DN = np.array([])
    for DN_values in data:
        interpol = np.interp(max_lowest_int_time, DN_values, intTimes_sorted)
        result = np.divide(DN_values, intTimes_sorted) * interpol / max_lowest_int_time  # Expected DN normalized to 1050 DN
        DN_nonlin = np.append(DN_nonlin, result)
        all_DN = np.append(all_DN, DN_values)
    sort_indx = all_DN.argsort()
    DN = all_DN[sort_indx]
    DN_non = DN_nonlin[sort_indx]
    def kernel(x, shift, sigma):
        return np.exp(-((x - shift) ** 2 / (2 * sigma ** 2)))

    averaging_DN = np.arange(min(DN), max(DN), gaussian_mean_steps)
    weighted_data = []
    for i in averaging_DN:
        weighted_data.append(np.sum(kernel(DN, i, sigma) * DN_non)/ np.sum(kernel(DN, i, sigma)))
    result = np.array(weighted_data)
    s3 = inter.UnivariateSpline(averaging_DN[index_start_spline_fit:], result[index_start_spline_fit:])
    nonlinear_factors = np.concatenate((result[0:index_start_spline_fit], s3(averaging_DN[index_start_spline_fit:])))

    plt.plot(DN, DN_non, '+')
    plt.plot(averaging_DN, nonlinear_factors, 'y')
    plt.title('Nonlinear Correction')
    plt.xlabel('DN')
    plt.ylabel('rel. error DN')
    plt.show()
    return averaging_DN, nonlinear_factors


def check_nonlinearity(cal_dict, correction_dict=None, min=0, max=-1, step=1):
    sorted_keys = sorted(cal_dict.keys())
    chosen_keys = [key for key in sorted_keys[min:max:step]]

    std_old = np.array([])

    import matplotlib.pyplot as plt
    for key in chosen_keys:
        plt.plot(cal_dict[key]['reference']['wave'], cal_dict[key]['reference']['mean'], label='%s' % key)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('DN')
    plt.legend()
    plt.show()

    gs = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    ax3 = plt.subplot(gs[2, :])
    for key in chosen_keys:
        # Darkcurrent subtraction
        darkcurrent_corrected_mean = (cal_dict[key]['reference']['mean'] - cal_dict[key]['darkcurrent']['mean']) / key
        ax1.plot(cal_dict[key]['reference']['wave'], darkcurrent_corrected_mean, label='%s' %key)
        try:
            std_old = np.vstack((std_old, darkcurrent_corrected_mean))
        except ValueError:
            std_old = darkcurrent_corrected_mean

    std_old_mean = np.mean(std_old, axis=0)
    std_old = np.std(std_old, axis=0)
    ax3.plot(cal_dict[key]['reference']['wave'], std_old / std_old_mean, label='Not corrected',color='r')

    if len(correction_dict) == 2: legends = ['Offset', 'Darkcurrent']
    if correction_dict:
        if type(correction_dict) is not list: correction_dict = [correction_dict]
        for i, corr_dict in enumerate(correction_dict):
            std_new = np.array([])
            for key in chosen_keys:
                calibrated_mean = (cal_dict[key]['reference']['mean']) / \
                                                 np.interp(cal_dict[key]['reference']['mean'], corr_dict['DN'], corr_dict['nonlinear'])
                # Darkcurrent subtraction
                calibrated_mean = (calibrated_mean - cal_dict[key]['darkcurrent']['mean'])/ key
                ax2.plot(cal_dict[key]['reference']['wave'], calibrated_mean, label='%s' %key)
                try:
                    std_new = np.vstack((std_new, calibrated_mean))
                except ValueError:
                    std_new = calibrated_mean
            std_mean = np.mean(std_new, axis=0)
            std_new = np.std(std_new, axis=0)
            ax3.plot(cal_dict[key]['reference']['wave'], std_new/std_mean, label='%s corrected' % legends[i])

    ax1.set_title('Not nonlinear corrected')
    ax2.set_title('Nonlinear corrected')
    ax1.set_ylabel('DN')
    ax2.set_ylabel('DN')
    ax3.set_xlabel('Wavelength [nm]')
    ax3.set_ylabel(r'$\Delta$ $\%$')
    ax3.legend(loc='best')
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()
    plt.setp(ltext, fontsize='small')
    plt.show()



