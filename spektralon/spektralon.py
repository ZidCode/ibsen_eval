import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties


FONTSTYLE = 'serif'
FONTSIZE = 12
hfont = {'family':FONTSTYLE, 'fontsize': FONTSIZE}
fontP = FontProperties()
fontP.set_family(FONTSTYLE)
fontP.set_size('small')


def scale_to_irradiance(ref):
    wave, specs = parse_spectralon()
    values = np.interp(ref['wave'], wave, specs)
    ref['mean'] = ref['mean'] / values * np.pi
    ref['tdata'] = np.divide(ref['tdata'], values) * np.pi
    ref['data'] = np.transpose(ref['tdata'])
    return ref


def parse_spectralon(file_ = os.path.dirname(os.path.realpath(__file__))+'/S1005_22590-41.txt'):
    data = np.genfromtxt(file_, skip_header=10)
    wave = data[:, 0]
    specs = data[:, 1]
    return wave, specs


def plot(wave, specs):
    plt.plot(wave, specs * 100)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Spectralon reflectance [$\%$]', **hfont)
    plt.show()


if __name__ == "__main__":
    wave, specs = parse_spectralon()
    plot(wave, specs)
    #ref = {'wave': np.linspace(350, 750, 1000), 'mean': np.linspace(0.01, 0.01, 1000)}
    #plt.plot(ref['wave'], ref['mean'])
    #ref = scale_to_irradiance(ref)
    #plt.plot(ref['wave'], ref['mean'])
    #plt.show()
