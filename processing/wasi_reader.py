import numpy as np
import os
from matplotlib.font_manager import FontProperties
from scipy.ndimage.filters import gaussian_filter


FONTSTYLE = 'serif'
FONTSIZE = 12
hfont = {'family':FONTSTYLE, 'fontsize': FONTSIZE}
fontP = FontProperties()
fontP.set_family(FONTSTYLE)
fontP.set_size('small')


def wasi_oxygen(filename=os.path.dirname(os.path.realpath(__file__)) + '/WASI_database/O2.A'):
    data = np.genfromtxt(filename, skip_header=4)
    wave_nm = data[:,0]
    O2 = data[:,1]
    return wave_nm, O2


def wasi_e0(filename=os.path.dirname(os.path.realpath(__file__)) + '/WASI_database/E0_sun.txt'):
    data = np.genfromtxt(filename, skip_header=11)
    wave_nm = data[:,0]
    e0 = data[:,1]
    e0 = gaussian_filter(e0, 0.5)
    return wave_nm, e0


def wasi_wv(filename=os.path.dirname(os.path.realpath(__file__)) + '/WASI_database/WV.A'):
    data = np.genfromtxt(filename, skip_header=4)
    wave_nm = data[:,0]
    wv = data[:,1]
    return wave_nm, wv


def wasi_ozone(filename=os.path.dirname(os.path.realpath(__file__)) + '/WASI_database/O3.A'):
    data = np.genfromtxt(filename, skip_header=4)
    wave_nm = data[:,0]
    O3 = data[:,1]
    return wave_nm, O3


def get_wasi_parameters():
    params = {'o2': wasi_oxygen, 'o3': wasi_ozone, 'e0': wasi_e0, 'wv': wasi_wv}
    wasi_dict = dict()
    for key, value in params.items():
        wasi_dict[key] = dict()
        wasi_dict[key]['wave'], wasi_dict[key]['values'] = value()
    return wasi_dict


def get_wasi(wave):
    w_oz, oz = wasi_ozone()
    ozone = np.interp(wave, w_oz, oz)
    w_o2, o2 = wasi_oxygen()
    oxygen = np.interp(wave, w_o2, o2)
    w_wv, wv = wasi_wv()
    water = np.interp(wave, w_wv, wv)
    w_eo, eo = wasi_e0()
    solar = np.interp(wave, w_eo, eo)
    return ozone, oxygen, water, solar


def plot():
    import matplotlib.pyplot as plt
    wv_o2, o2 = wasi_oxygen()
    wv_e0, e0 = wasi_e0()
    wv_o3, o3 = wasi_ozone()
    wv_wv, wv = wasi_wv()
    plt.plot(wv_o2, o2, label=r'Oxygen $O^2$')
    plt.xlabel("Wavelength [nm]", **hfont)
    plt.ylabel(r"mean absorption coefficient $\left[cm^{-1}\right]$", **hfont)
    plt.legend(loc='best', prop=fontP)
    plt.yscale('log')
    plt.show()
    plt.plot(wv_e0, e0)
    plt.xlabel("Wavelength [nm]", **hfont)
    plt.ylabel(r"mean extraterrestrial irradiance $H_0$ $\left[\frac{mW}{m^2\cdot nm}\right]$", **hfont)
    plt.legend(loc='best', prop=fontP)
    plt.show()
    plt.plot(wv_o3, o3, label=r'Ozone $O^3$')
    plt.xlabel("Wavelength [nm]", **hfont)
    plt.ylabel(r"mean absorption coefficient $\left[cm^{-1}\right]$", **hfont)
    plt.legend(loc='best', prop=fontP)
    plt.yscale('log')
    plt.show()
    plt.plot(wv_wv, wv, label=r'Water Vapor')
    plt.ylabel(r"mean absorption coefficient $\left[cm^{-1}\right]$", **hfont)
    plt.xlabel("Wavelength [nm]", **hfont)
    plt.legend(loc='best', prop=fontP)
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    plot()
    get_wasi_parameters()
