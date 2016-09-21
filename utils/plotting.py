import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_meas(tar, ref, dark):
    gs = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    ax3 = plt.subplot(gs[2, :])
    map_dicts = {'Target_E_ds': {'plt': ax1, 'meas': tar},
                 'Reference_E_dd': {'plt': ax2, 'meas': ref},
                 'Darkcurrent': {'plt': ax3, 'meas':dark}}

    for k, v in map_dicts.items():
        v['plt'].plot(v['meas']['wave'], v['meas']['data'])
        v['plt'].set_title('%s' % k)

    plt.tight_layout()
    plt.xlabel('Wavelength [nm]')
    plt.show()


def plot_used_irradiance_and_reflectance(tarmd, refmd, reflectance):
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    map_dict = {'E_ds_mean': tarmd, 'E_dd_mean': refmd}

    for key, meas in map_dict.items():
        ax1.plot(meas['wave'], meas['data'], alpha=0.1)
        ax1.plot(meas['wave'], meas['mean'], label='%s' %key)

    ax1.set_title('Diffuse and direct Irradiance')
    ax2.plot(reflectance['wave_mu'], reflectance['spectra'], '+')
    ax2.errorbar(reflectance['wave_mu'], reflectance['spectra'], yerr=reflectance['std'], ecolor='g', fmt='none')
    ax2.set_title('Reflectance')
    ax1.legend()
    plt.xlabel('Wavelength [nm]')
    plt.show()


def plot_fitted_reflectance(reflectance_dict, params, result):
    """
    Args:
        reflectance_dict: Reflectance dict with std, wave and spectra
        params: Lmfit fitted values for corresponding spectra
        result: Lmfit MinimizerResult() object
    """
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])

    ax1 = result.plot_residuals(ax=ax1)
    ax2.plot(reflectance_dict['wave_mu'], reflectance_dict['spectra'])
    ax2.plot(params['wave_range'], result.best_fit, 'r-')
    ax2.errorbar(params['wave_range'], params['spectra_range'], yerr=params['std'], ecolor='g')
    ax2.set_title('Fitted reflectance')
    ax2.set_ylabel('Reflectance')
    ax2.set_xlabel(r'Wavelength $\left[\mu m\right]$')
    plt.show()


def plot_factory(ax1, ax2, param):
    if 'alpha_stderr' in param:
        ax1.errorbar(param['utc_times'], param['alpha'], yerr=param['alpha_stderr'], ecolor='g', fmt='none',label=param['label'])
    else:
        ax1.plot(param['utc_times'], param['alpha'], 'r+', label=param['label'])
    if 'beta_stderr' in param:
        ax2.errorbar(param['utc_times'], param['beta'], yerr=param['beta_stderr'], ecolor='g', fmt='none',label=param['label'])
    else:
        ax2.plot(param['utc_times'], param['beta'], 'r+', label=param['label'])
    return ax1, ax2


def plot_aengstrom_parameters(*param_dict):

    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])

    for param in param_dict:
        ax1, ax2 = plot_factory(ax1, ax2, param)

    ax1.set_title('Aengstrom parameters')
    ax1.set_ylabel(r'Aengstrom $\alpha$')
    ax2.set_ylabel(r'Aengstrom $\beta$')
    ax2.set_xlabel('UTC Time')
    ax1.legend(loc='best')
    plt.show()
