from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import pandas as pd
from matplotlib.font_manager import FontProperties
from utils.TitleDecorator import TitleCreator


FONTSTYLE = 'serif'
FONTSIZE = 12
hfont = {'family':FONTSTYLE, 'fontsize': FONTSIZE}
fontP = FontProperties()
fontP.set_family(FONTSTYLE)
fontP.set_size('small')


class Absolute:
    def __init__(self, result, _):
        self.result = result

    def __call__(self):
        return self.result


class Relative:
    def __init__(self, result, expect):
        self.result = result
        self.expect = expect

    def __call__(self):
        return self.result / self.expect * 100


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


LableTable = {
    'beta': r'absolute $\Delta \beta \;$',
    'alpha': r'$\Delta \alpha \; \left[ \%\right]$',
    'g_dsa': r'Coverty factor $\Delta$ $g_{dsa} \left[ \%\right]$',
    'g_dsr': r'Coverty factor $\Delta$ $g_{dsr} \left[ \%\right]$',
    'l_dsa': r'Intensity factor $\Delta$ $l_{dsa} \left[ \%\right]$',
    'l_dsr': r'Intensity factor $\Delta$ $l_{dsr} \left[ \%\right]$',
    'wv': r'$\Delta$ $Water \, Vapour \left[ \%\right]$',
    'H_oz': r'Ozone scale height $\Delta$ $H_{oz} \left[ \%\right]$'
}


def plot_2D(x_new, y_new, z_new, label, title, config):
    plt.figure()
    plt.pcolormesh(x_new, y_new, z_new, norm=MidpointNormalize(midpoint=0.), cmap='RdBu_r')
    plt.gca().set_aspect("auto")
    cb = plt.colorbar(label=label)
    CS = plt.contour(x_new, y_new, z_new, colors='k')
    plt.clabel(CS, inline=1, fontsize=10, fmt='%1.2f')
    plt.xlabel(LableTable[config['local']], **hfont)
    plt.ylabel(LableTable[config['global']], **hfont)
    plt.title('%s' %title)
    ax = cb.ax
    text = ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(family='serif', size=12)
    text.set_font_properties(font)
    plt.show()


def show_result(result, config):
    idx = result['idx']
    expect = result['expected']
    builder = TitleCreator(config['keys'], result['expected'])
    title = builder.built_title()

    for key in config['show']:
        key_result = config['present'][key](result[key], expect[idx[key]])
        plot_2D(result[config['local']] / expect[idx[config['local']]] * 100, result[config['global']] / expect[idx[config['global']]] * 100,
                key_result() , LableTable[key], title, config)


def read_twoD(dir, counts, config):
    files_ = [dir + 'two_variation_Deltaoutput_%s.txt' % i for i in range(counts)]
    result_dict = {key:0 for key in config['keys']}
    result_dict['idx'] = {key:idx for idx, key in enumerate(config['keys'])}
    frame = pd.read_csv(files_[0])
    for param in config['show']:
        result_dict[param] = frame['%s_mean' % param]
    result_dict[config['local']] = frame[config['local']]
    # Get first parameter
    result_dict[config['global']] = frame[config['global']][0]
    result_dict['expected'] = frame['expected'][0:len(config['keys'])]
    for file_ in files_[1:]:
        frame = pd.read_csv(file_)
        for param in config['show']:
            result_dict[param] = np.vstack((result_dict[param], frame['%s_mean' % param]))
        result_dict[config['global']] = np.append(result_dict[config['global']], frame[config['global']][0])
    return result_dict


def main(dir, counts, config):
    result = read_twoD(dir, counts, config)
    show_result(result, config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='/home/jana_jo/DLR/Codes/evaluation/processing/results/twoD_1p8_0p06_0p5_0p8/')
    parser.add_argument('-c', '--counts', default=41, type=int)
    config = dict()
    config['keys'] = ['alpha', 'beta', 'l_dsr', 'l_dsa', 'H_oz', 'wv']
    config['local'] = 'wv'
    config['global'] = 'H_oz'
    config['show'] =  ['l_dsr', 'beta']
    config['present'] = {'l_dsr': Relative, 'beta': Absolute}
    args = parser.parse_args()
    main(args.directory, args.counts, config)
