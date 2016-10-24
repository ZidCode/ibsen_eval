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
    #ax2.errorbar(reflectance['wave_mu'], reflectance['spectra'], yerr=reflectance['std'], ecolor='g', fmt='none')
    ax2.set_title('Reflectance')
    ax1.legend()
    plt.xlabel('Wavelength [nm]')
    plt.show()


def plot_fitted_reflectance(aero_fit):
    """
    Args:
        reflectance_dict: Reflectance dict with std, wave and spectra
        params: Lmfit fitted values for corresponding spectra
        result: Lmfit MinimizerResult() object
    """
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])

    ax1 = aero_fit.result.plot_residuals(ax=ax1)
    ax2.plot(aero_fit.spectra['wave_mu'], aero_fit.spectra['spectra'])
    ax2.plot(aero_fit.param_dict['wave_range'], aero_fit.result.best_fit, 'r-')
    ax2.errorbar(aero_fit.param_dict['wave_range'], aero_fit.param_dict['spectra_range'], yerr=aero_fit.param_dict['std'], ecolor='g')
    ax2.set_title('Fitted reflectance')
    ax2.set_ylabel('Reflectance')
    ax2.set_xlabel(r'Wavelength $\left[\mu m\right]$')
    plt.show()

def get_ax(ax, param, param_key, col):
    #ax.plot(param['utc_times'], param['%s' % param_key], '%s' %col, label=param['label'])
    ax.errorbar(param['utc_times'], param['%s' % param_key], yerr=param['%s_stderr' % param_key], ecolor='%s' % col, fmt='none',label=param['label'])
    return ax


def plot_factory(ax1, ax2, param):
    if param['label'] == 'microtops':
        ax1 = get_ax(ax1, param, 'alpha', 'r')
        ax2 =get_ax(ax2, param, 'beta', 'r')
    elif param['label'] == 'Ibsen':
        ax1 = get_ax(ax1, param, 'alpha', 'g')
        ax2 = get_ax(ax2, param, 'beta', 'g')
    return ax1, ax2


def plot_aengstrom_parameters(*param_dict):

    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])

    for param in param_dict:
        ax1, ax2 = plot_factory(ax1, ax2, param)

    ax1.set_ylabel(r'Aengstrom exponent $\alpha$')
    ax2.set_ylabel(r'Turbidity $\beta$')
    ax2.set_xlabel('UTC Time')
    ax1.legend(loc='best')
    plt.show()
