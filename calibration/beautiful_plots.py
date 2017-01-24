import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


FONTSTYLE = 'serif'
FONTSIZE = 12

hfont = {'family':FONTSTYLE, 'fontsize': FONTSIZE}

x = np.linspace(0, 10, 100)
y = np.linspace(23, 111, 100)

fontP = FontProperties()
fontP.set_family(FONTSTYLE)
fontP.set_size('small')


plt.plot(x, y, label='first')
plt.plot(x, y, label='second')

plt.xlabel(r'Wavelength $\lambda$ [nm]', **hfont)
plt.ylabel(r'DN [a.u.]', **hfont)

legend = plt.legend(loc=0, ncol=1, prop = fontP,fancybox=True,
                    shadow=False,title='Integration Times [ms]',bbox_to_anchor=(1.0, 1.0))
plt.setp(legend.get_title(),fontsize='small', family=FONTSTYLE)
plt.tight_layout()
plt.show()


gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :])

ax1.plot(x,y, label='up')
ax2.plot(x, y, label='down')
ax1.set_xlabel('Integration Time [ms]', **hfont)
ax2.set_xlabel('Wavelength [nm]', **hfont)
ax1.set_ylabel('DN [a.u.]', **hfont)
ax1.legend(loc='best', prop=fontP, title='Test')
ax2.legend(loc='best', prop=fontP)
plt.setp(ax1.get_legend().get_title(),fontsize='small', family=FONTSTYLE)
plt.tight_layout()
plt.show()


