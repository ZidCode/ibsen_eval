import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties



FONTSTYLE = 'serif'
FONTSIZE = 12
hfont = {'family':FONTSTYLE, 'fontsize': FONTSIZE}
fontP = FontProperties()
fontP.set_family(FONTSTYLE)
fontP.set_size('small')


def plot_meas(tar, ref):
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    map_dicts = {r'Radiance $E_{ds}$ Zenith': {'plt': ax1, 'meas': tar},
                 r'Irradiance $E_d$': {'plt': ax2, 'meas': ref}}

    for k, v in map_dicts.items():
        v['plt'].plot(v['meas']['wave'], v['meas']['data'])
        v['plt'].set_title('%s' % k, **hfont)


    plt.xlabel('Wavelength [nm]', **hfont)
    ax1.set_ylabel(r'$\frac{mW}{nm m^2 sr}$', **hfont)
    ax2.set_ylabel(r'$\frac{mW}{nm m^2}$', **hfont)
    plt.tight_layout()
    plt.show()


def plot_used_irradiance_and_reflectance(tarmd, refmd, reflectance):
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    map_dict = {'E_ds_mean': tarmd, 'E_d_mean': refmd}

    for key, meas in map_dict.items():
        ax1.plot(meas['wave'], meas['data'], alpha=0.1)
        ax1.plot(meas['wave'], meas['mean'], label='%s' %key)

    ax1.set_title(r'$E_{ds}$ and $E_{d}$', **hfont)
    ax2.plot(reflectance['wave_nm'], reflectance['spectra'], '+')
    ax1.set_ylabel(r'$\frac{mW}{nm m^2}$', **hfont)
    ax2.set_ylabel(r'Ratio [$\%$]', **hfont)
    ax1.legend(loc='best', prop=fontP)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.show()


def plot_fitted_reflectance(result, param_dict, measurement):
    """$
    Args:
        reflectance_dict: Reflectance dict with std, wave and spectra
        params: Lmfit fitted values for corresponding spectra
        result: Lmfit MinimizerResult() object
    """
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])

    ax1 = result.plot_residuals(ax=ax1)
    ax2.plot(measurement['wave_nm'], measurement['spectra'])
    ax2.plot(param_dict['wave_range'], result.best_fit, 'r-')
    ax2.errorbar(param_dict['wave_range'], param_dict['spectra_range'], yerr=param_dict['std'], ecolor='g')
    ax2.set_title('Fitted ratio', **hfont)
    ax2.set_ylabel('Ratio', **hfont)
    ax2.set_xlabel(r'Wavelength $\left[nm\right]$', **hfont)
    plt.show()


def plot_aengstrom_parameters(results, validation, title):

    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    ax1.errorbar(results['sun_zenith'], results['alpha'], yerr=results['alpha_stderr'], ecolor='b',label='Ibsen')
    ax1.errorbar(validation['utc_times'], validation['alpha'], yerr=validation['alpha_stderr'], ecolor='g',label='Microtops')
    ax2.errorbar(results['sun_zenith'], results['beta'], yerr=results['beta_stderr'], ecolor='b' ,label='Ibsen')
    ax2.errorbar(validation['utc_times'], validation['beta'], yerr=validation['beta_stderr'], ecolor='g',label='Microtops')
    ax1.legend(loc='best', prop=fontP)
    ax2.legend(loc='best', prop=fontP)
    ax1.set_ylabel(r'Aengstrom exponent $\alpha$', **hfont)
    ax2.set_ylabel(r'Turbidity $\beta$', **hfont)
    ax2.set_xlabel('UTC Time', **hfont)
    ax1.set_title('%s' % title, **hfont)
    plt.tight_layout()
    plt.show()


def plot_turbidity(wave, aods, wave_new, fitted):
    plt.plot(wave, aods, 'b+')
    plt.plot(wave_new, fitted, 'g')
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'AOD', **hfont)
    return plt


def plot_aengstrom_parameters_aeronet(object_list, title):

    fig = plt.figure()
    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    ax3 = plt.subplot(gs[2, :])
    ax4 = plt.subplot(gs[3, :])
    for obj in object_list:
        obj.get_plot([ax1, ax2, ax3, ax4])

    ax1.legend(loc='best', prop=fontP)
    ax2.legend(loc='best', prop=fontP)
    ax3.legend(loc='best', prop=fontP)
    ax4.legend(loc='best', prop=fontP)

    ax1.set_ylabel(r'Aengstrom exponent $\alpha$', **hfont)
    ax2.set_ylabel(r'Turbidity $\beta$', **hfont)
    ax3.set_ylabel(r'$g_{dsr}$', **hfont)
    ax4.set_ylabel(r'$g_{dsa}$', **hfont)
    ax4.set_xlabel('UTC Times', **hfont)
    ax1.set_title('%s' % title, **hfont)
    plt.tight_layout()
    plt.show()


def ibsen_plot(frame, ax1, ax2, ax3, ax4):
    ax1.errorbar(frame['utc_times'], frame['alpha'], yerr=frame['alpha_stderr'],fmt='o',
                 label='Ibsen', markersize='2', color='b', ecolor='b')
    ax2.errorbar(frame['utc_times'], frame['beta'], yerr=frame['beta_stderr'], label='Ibsen', fmt='o', markersize='2', color='b', ecolor='b')
    ax3.errorbar(frame['utc_times'], frame['g_dsr'], yerr=frame['g_dsr_stderr'],
                 label='Ibsen')
    ax4.errorbar(frame['utc_times'], frame['g_dsa'], yerr=frame['g_dsa_stderr'],
                 label='Ibsen')
    return ax1, ax2, ax3, ax4


def aeronet_plot(aeronet, ax1, ax2, _, __):
    ax1.plot(aeronet['utc_times'], aeronet['440-870_Angstrom_Exponent'], '+', label='440-870')
    ax1.plot(aeronet['utc_times'], aeronet['380-500_Angstrom_Exponent'], '+', label='380-500')
    ax1.plot(aeronet['utc_times'], aeronet['440-675_Angstrom_Exponent'], '+', label='440-675')
    ax1.plot(aeronet['utc_times'], aeronet['500-870_Angstrom_Exponent'], '+', label='500-870')
    ax2.errorbar(aeronet['utc_times'], aeronet['Turbidity'], yerr=aeronet['Turbidity_stderror'],fmt='o',
                 label='Aeronet', color='r', ecolor='r', markersize='2')

    return ax1


def micro_plot(micro_dict, ax1, ax2, _, __):
    ax1.errorbar(micro_dict['utc_times'], micro_dict['alpha'], yerr=micro_dict['alpha_stderr'],
                 label='Microtops', fmt='o', markersize='2', color='g', ecolor='g')
    ax2.errorbar(micro_dict['utc_times'], micro_dict['beta'], yerr=micro_dict['beta_stderr'],
                 label='Microtops', fmt='o', markersize='2', color='g', ecolor='g')
    return ax1, ax2
