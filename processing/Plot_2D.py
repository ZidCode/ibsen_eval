from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import pandas as pd
from matplotlib.font_manager import FontProperties


FONTSTYLE = 'serif'
FONTSIZE = 12
hfont = {'family':FONTSTYLE, 'fontsize': FONTSIZE}
fontP = FontProperties()
fontP.set_family(FONTSTYLE)
fontP.set_size('small')


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_2D(x_new, y_new, z_new, label, title):
    plt.figure()
    plt.pcolormesh(x_new, y_new, z_new, norm=MidpointNormalize(midpoint=0.), cmap='RdBu_r')
    plt.gca().set_aspect("auto")
    cb = plt.colorbar(label=label)
    CS = plt.contour(x_new, y_new, z_new, colors='k')
    plt.clabel(CS, inline=1, fontsize=10, fmt='%1.2f')
    plt.xlabel(r'Coverty factor $\Delta$ $g_{dsa} \left[ \%\right]$', **hfont)
    plt.ylabel(r'Coverty factor $\Delta$ $g_{dsr} \left[ \%\right]$', **hfont)
    plt.title('%s' %title)
    ax = cb.ax
    text = ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(family='serif', size=12)
    text.set_font_properties(font)
    plt.show()


def show_result(result):
    idx = result['keys']
    expect = result['expected']
    title = r'$\alpha=%s$, $\beta=%s$, $g_{dsa}=%s$, $g_{dsr}=%s$' % (expect[idx['alpha']], expect[idx['beta']], expect[idx['g_dsa']], expect[idx['g_dsr']])
    label = r'$\Delta \alpha \; \left[ \%\right]$'
    plot_2D(result['g_dsa'] / expect[idx['g_dsa']] * 100, result['g_dsr'] / expect[idx['g_dsr']] * 100, result['alpha'] /expect[idx['alpha']] * 100, label, title)

    label = r'absolute $\Delta \beta \;$'
    plot_2D(result['g_dsa'] / expect[idx['g_dsa']] * 100, result['g_dsr'] / expect[idx['g_dsr']] * 100, result['beta'], label, title)


def read_twoD(dir, counts, keys):
    files_ = [dir + 'two_variation_Deltaoutput_%s.txt' % i for i in range(counts)]
    result_dict = {key:0 for key in keys}
    result_dict['keys'] = {key:idx for idx, key in enumerate(keys)}
    frame = pd.read_csv(files_[0])

    result_dict['g_dsa'] = frame['g_dsa']
    result_dict['g_dsr'] = frame['g_dsr'][0]
    result_dict['alpha'] = frame['alpha_mean']
    result_dict['beta'] = frame['beta_mean']
    result_dict['expected'] = frame['expected'][0:len(keys)]

    for file_ in files_[1:]:
        frame = pd.read_csv(file_)
        result_dict['alpha'] = np.vstack((result_dict['alpha'], frame['alpha_mean']))
        result_dict['beta'] = np.vstack((result_dict['beta'], frame['beta_mean']))
        result_dict['g_dsr'] = np.append(result_dict['g_dsr'], frame['g_dsr'][0])
    return result_dict


def main(dir, counts, keys):
    result = read_twoD(dir, counts, keys)
    show_result(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='/home/jana_jo/DLR/Codes/evaluation/processing/results/twoD_1p8_0p06_0p5_0p8/')
    parser.add_argument('-c', '--counts', default=41, type=int)
    keys = ['alpha', 'beta', 'g_dsa', 'g_dsr']
    args = parser.parse_args()
    main(args.directory, args.counts, keys)