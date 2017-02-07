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
    map_dicts = {r'Radiance $L_{sky}$ Zenith': {'plt': ax1, 'meas': tar},
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
    map_dict = {'L_sky_mean': tarmd, 'E_d_mean': refmd}

    for key, meas in map_dict.items():
        ax1.plot(meas['wave'], meas['data'], alpha=0.1)
        ax1.plot(meas['wave'], meas['mean'], label='%s' %key)

    ax1.set_title(r'$L_{sky}$ and $E_{d}$', **hfont)
    ax2.plot(reflectance['wave_nm'], reflectance['spectra'], '+')
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
    ax2.set_title('Fitted reflectance', **hfont)
    ax2.set_ylabel('Reflectance', **hfont)
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


def plot_aengstrom_parameters_aeronet(results, validation, micro, title):

    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    ax3 = plt.subplot(gs[2, :])
    ax4 = plt.subplot(gs[3, :])
    ax1.errorbar(results['utc_times'], results['alpha'], yerr=results['alpha_stderr'], ecolor='b',label='Ibsen')
    ax1.plot(validation['utc_times'], validation['440-870_Angstrom_Exponent'], '+', label='440-870')
    ax1.plot(validation['utc_times'], validation['380-500_Angstrom_Exponent'], '+', label='380-500')
    ax1.plot(validation['utc_times'], validation['440-675_Angstrom_Exponent'], '+', label='440-675')
    ax1.plot(validation['utc_times'], validation['500-870_Angstrom_Exponent'], '+', label='500-870')
    ax1.errorbar(micro['utc_times'], micro['alpha'], yerr=micro['alpha_sterr'], ecolor='g', label='Microtops')

    ax2.errorbar(results['utc_times'], results['beta'], yerr=results['beta_stderr'], ecolor='b' ,label='Ibsen')
    ax2.errorbar(validation['utc_times'], validation['Turbidity'], yerr=validation['Turbidity_stderror'], fmt='None', ecolor='g', label='Aeronet')
    ax2.errorbar(micro['utc_times'], micro['beta'], yerr=micro['beta_stderr'], ecolor='r' ,label='Microtops')

    ax3.errorbar(results['utc_times'], results['g_dsr'], yerr=results['g_dsr_stderr'], ecolor='b',label='Ibsen')
    ax4.errorbar(results['utc_times'], results['g_dsa'], yerr=results['g_dsa_stderr'], ecolor='b', label='Ibsen')

    ax1.legend(loc='best', prop=fontP)
    ax2.legend(loc='best', prop=fontP)
    ax3.legend(loc='best', prop=fontP)
    ax4.legend(loc='best', prop=fontP)

    ax1.set_ylabel(r'Aengstrom exponent $\alpha$', **hfont)
    ax2.set_ylabel(r'Turbidity $\beta$', **hfont)
    ax3.set_ylabel(r'$g_{dsr}$', **hfont)
    ax4.set_ylabel(r'$g_{dsa}$', **hfont)
    ax2.set_xlabel('Sun zenith $^{\circ}$', **hfont)
    ax1.set_title('%s' % title, **hfont)
    plt.tight_layout()
    plt.show()


def plot_turbidity(wave, aods, wave_new, fitted):
    plt.plot(wave, aods, 'b+')
    plt.plot(wave_new, fitted, 'g')
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'AOD', **hfont)
    return plt
