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
    ax2.plot(tarmd['wave'], reflectance)
    ax2.set_title('Reflectance')
    ax1.legend()
    plt.xlabel('Wavelength [nm]')
    plt.show()



