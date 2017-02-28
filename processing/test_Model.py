import theano
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from BaseModels import BaseModelPython, BaseModelSym
from Model import SkyRadiance, WaterVapourTransmittance, OzoneTransmittance, SkyRadianceSym, IrradianceRatio, IrradianceRatioSym, LSkyRatioSym, LSkyRatio
from FitModel import FitWrapper, Minimize
from Residuum import Residuum
from matplotlib.font_manager import FontProperties


FONTSTYLE = 'serif'
FONTSIZE = 12
hfont = {'family':FONTSTYLE, 'fontsize': FONTSIZE}
fontP = FontProperties()
fontP.set_family(FONTSTYLE)
fontP.set_size('small')


def test_main():
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 950
    AM = 5
    ssa = get_ssa(rel_h, AM)
    x = np.linspace(350, 800, 100)  # config
    variables = ['alpha', 'beta', 'g_dsa', 'g_dsr']  # config
    expected_values = [2.5, 0.06, 0.6, 0.8]

    print('Expected: %s' % expected_values)
    guess = [1.0, 0.01, 0.5, 0.7]  # config
    bounds = [(-0.2, 4), (0., 3), (0., 2.), (0., 2.)]  # config

    # Theano
    irr_symbol = IrradianceRatioSym(zenith, AM, pressure, ssa, x, variables)
    getIrrRatio = irr_symbol.get_compiled()
    y_theano = getIrrRatio(*expected_values)
    res = Residuum(irr_symbol)
    residuum = FitWrapper(res.getResiduum())


    # Python
    kwargs = {key:value for key,value in zip(variables, expected_values)}
    print("parameters :  %s" % kwargs)
    bm = BaseModelPython(zenith, AM, pressure, ssa)
    irr_python = IrradianceRatio(bm)
    y_python = irr_python.func(x, **kwargs)

    gmod = Model(irr_python.func, independent_vars=['x'], param_names=variables)
    gmod.set_param_hint('alpha', value=guess[0], min=bounds[0][0], max=bounds[0][1])
    gmod.set_param_hint('beta',  value=guess[1], min=bounds[1][0], max=bounds[1][1])
    gmod.set_param_hint('g_dsa', value=guess[2], min=bounds[2][0], max=bounds[2][1])
    gmod.set_param_hint('g_dsr', value=guess[3], min=bounds[3][0], max=bounds[3][1])

    result_lmfit = gmod.fit(y_python, x=x)
    print(result_lmfit.fit_report())

    plt.plot(x, y_theano)
    x_new = np.linspace(300, 900,150)
    irr_symbol.set_wavelengthAOI(x_new)
    getIrrRatio = irr_symbol.get_compiled()
    y_new = getIrrRatio(*expected_values)
    plt.plot(x_new, y_new, '+', label='different wavelengths')
    plt.legend()
    plt.show()


def sky_radiance():
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 950
    AM = 5
    ssa = get_ssa(rel_h, AM)
    x = np.linspace(780, 850, 1000)  # config
    H_oz = 0.3
    wv = 0.25
    alpha = 1.6
    beta = 0.06
    l_dsr = 0.02
    l_dsa = 0.02

    model = BaseModelPython(zenith, AM, pressure, ssa)
    skyModel = SkyRadiance(model)

    ozone = OzoneTransmittance(model)
    oz = ozone.func(x, H_oz)
    oz2 = ozone.func(x, 0.33)
    oz3 = ozone.func(x, 0.34)

    ww = WaterVapourTransmittance(model)
    water = ww.func(0.25, x)
    water2 = ww.func(1.5, x)

    for wv in np.arange(0.2, 2.5, 0.1):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, wv=wv, H_oz=H_oz)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("Water Vapour 0.2-2.5 cm", **hfont)
    plt.show()

    for hoz in np.arange(0.3, 0.7, 0.1):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, wv=wv, H_oz=hoz)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("Ozone 0.3-0.7 cm", **hfont)
    plt.show()

    for l in np.arange(0.005, 0.1, 0.01):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l, wv=wv, H_oz=hoz)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("l_dsr 0.005-0.1", **hfont)
    plt.show()

    for l in np.arange(0.005, 0.1, 0.01):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l, l_dsr=l_dsr, wv=wv, H_oz=hoz)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("l_dsa 0.005-0.1", **hfont)
    plt.show()

    for a in np.arange(1.4, 1.8, 0.01):
        y = skyModel.func(x=x, alpha=a, beta=beta, l_dsa=l, l_dsr=l_dsr, wv=wv, H_oz=hoz)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("alpha 1.4-1.8", **hfont)
    plt.show()

    for b in np.arange(0.02, 0.1, 0.01):
        y = skyModel.func(x=x, alpha=alpha, beta=b, l_dsa=l, l_dsr=l_dsr, wv=wv, H_oz=hoz)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("beta 0.02-0.1", **hfont)
    plt.show()

    plt.plot(x, water, label="WV 0.25cm")
    plt.plot(x, water2, label="WV 1.5cm")
    plt.plot(x,oz, label="Hoz 0.3cm")
    plt.plot(x,oz2, label="Hoz 0.33cm")
    plt.plot(x,oz3, label="Hoz 0.34cm")
    plt.legend(loc='best', prop=fontP)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel('Transmittance', **hfont)
    plt.show()


def l_sky_ratio():
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 950
    AM = 5
    ssa = get_ssa(rel_h, AM)
    x = np.linspace(350, 700, 1000)  # config
    alpha = 1.6
    beta = 0.06
    l_dsr = 0.02
    l_dsa = 0.02
    g_dsr = 0.9
    g_dsa = 0.9

    model = BaseModelPython(zenith, AMass, pressure, ssa)
    skyModel = LSkyRatio(model)

    for l_dsr in np.arange(0.01, 0.1, 0.001):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("l_dsr 0.01-0.1", **hfont)
    plt.show()

    for l_dsa in np.arange(0.01, 0.1, 0.001) :
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("l_dsa 0.01-0.1", **hfont)
    plt.show()

    for g_dsr in np.arange(0.7, 0.99, 0.01):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("g_dsr 0.7-0.99", **hfont)
    plt.show()

    for g_dsa in np.arange(0.7, 0.99, 0.01):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr,g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("g_dsa 0.7-0.99", **hfont)
    plt.show()

    for a in np.arange(1.4, 1.8, 0.01):
        y = skyModel.func(x=x, alpha=a, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr,g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("alpha 1.4-1.8", **hfont)
    plt.show()

    for b in np.arange(0.02, 0.1, 0.001):
        y = skyModel.func(x=x, alpha=alpha, beta=b, l_dsa=l_dsa, l_dsr=l_dsr,g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("beta 0.02-0.1", **hfont)
    plt.show()


def compare_sym_python():
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 950
    AM = 5
    ssa = get_ssa(rel_h, AM)
    x = np.linspace(350, 800, 1000)  # config
    alpha = 1.8
    beta = 0.06
    l_dsr = 0.1
    l_dsa = 0.05
    g_dsr = 0.8
    g_dsa = 0.5
    model = BaseModelPython(zenith, AM, pressure, ssa)
    skyModel = LSkyRatio(model)
    y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, g_dsr=g_dsr, g_dsa=g_dsa,)

    symmodel = LSkyRatioSym(zenith, AM, pressure, ssa, x, ['alpha', 'beta','g_dsr', 'g_dsa', 'l_dsr', 'l_dsa'])
    func = symmodel.get_compiled()
    ysym = func(alpha, beta, g_dsr, g_dsa, l_dsr, l_dsa)

    plt.plot(x, ysym, label='sym')
    plt.plot(x, y, label='python')
    plt.legend()
    plt.show()


def fit_skyRadiance():
    from scipy.optimize import minimize
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 950
    AM = 5
    ssa = get_ssa(rel_h, AM)
    x = np.linspace(780, 840, 1000)  # config
    H_oz = 0.34
    wv = 1.2
    alpha = 1.8
    beta = 0.06
    l_dsr = 0.02
    l_dsa = 0.02
    guess = [alpha-0.2, beta+0.02, l_dsr-0.1, l_dsa+0.1]  # config
    bounds = [(-0.2, 3), (0.01, 0.3), (-0.5, 0.5), (-0.5, 0.5)]  # config

    symmodel = SkyRadianceSym(zenith, AM, pressure, ssa, x, ['alpha', 'beta', 'l_dsr', 'l_dsa', 'H_oz', 'wv'])
    func = symmodel.get_compiled()
    simulation = func(alpha, beta, l_dsr, l_dsa, H_oz, wv)
    symmodel.setVariable('H_oz', H_oz+0.4)
    symmodel.setVariable('wv', wv)
    print(symmodel.get_Symbols())
    res = Residuum(symmodel)
    callable = FitWrapper(res.getResiduum())
    derivative = FitWrapper(res.getDerivative())
    print(symmodel.getVariable('H_oz'))
    result = minimize(callable, guess, args=(simulation), jac=derivative, method='TNC', bounds=bounds)
    print(result.x)
    plt.plot(x, simulation, label='sym')
    func = symmodel.get_compiled()
    fitted = func(*result.x)
    plt.plot(x, fitted)
    plt.show()


def coverty_variability():
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 950
    AM = 5
    ssa = get_ssa(rel_h, AM)
    x = np.linspace(350, 800, 1000)  # config
    H_oz = 0.3
    wv = 0.25
    alpha = 1.8
    beta = 0.06
    l_dsr = 0.02
    l_dsa = 0.02
    #l_dsa = np.linspace(0.01, 0.1, 10)
    #for l in l_dsa:

    symmodel = SkyRadianceSym(zenith, AM, pressure, ssa, x, ['alpha', 'beta', 'l_dsr', 'l_dsa', 'H_oz', 'wv'])
    func = symmodel.get_compiled()
    ysym = func(alpha, beta, l_dsr, l_dsa, H_oz, wv)
    plt.plot(x, ysym, label='sym')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    #test_main()
    #sky_radiance()
    #compare_sym_python()
    #coverty_variability()
    fit_skyRadiance()
