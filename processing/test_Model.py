import theano
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from BaseModels import BaseModelPython, BaseModelSym
from Model import SkyRadiance, WaterVapourTransmittance, OzoneTransmittance, SkyRadianceSym, IrradianceRatio, IrradianceRatioSym, LSkyRatioSym, LSkyRatio
from FitModel import FitWrapper, Minimize
from Residuum import Residuum
from matplotlib.font_manager import FontProperties
from atmospheric_mass import get_atmospheric_path_length


FONTSTYLE = 'serif'
FONTSIZE = 20
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
    print(ssa)
    rel_h = 0.8
    ssa = get_ssa(rel_h, AM)
    print(ssa)
    x = np.linspace(350, 800, 100)  # config
    variables = ['alpha', 'beta', 'g_dsa', 'g_dsr']  # config
    expected_values = [2.5, 0.06, 0.6, 0.8]

    print('Expected: %s' % expected_values)
    guess = [2.3, 0.05, 0.5, 0.7]  # config
    bounds = [(-0.2, 4), (0., 3), (0., 2.), (0., 2.)]  # config

    # Theano
    #irr_symbol = IrradianceRatioSym(zenith, pressure, ssa, x, variables)
    #getIrrRatio = irr_symbol.get_compiled()
    #y_theano = getIrrRatio(*expected_values) + np.random.normal(0, 0.001, len(x))
    #res = Residuum(irr_symbol)
    #residuum = FitWrapper(res.getResiduum())


    # Python
    kwargs = {key:value for key,value in zip(variables, expected_values)}
    print("parameters :  %s" % kwargs)
    bm = BaseModelPython(zenith, pressure, ssa)
    irr_python = IrradianceRatio(bm, bm)
    y_python = irr_python.func(x, **kwargs)# + np.random.normal(0, 0.0005, len(x))


    gmod = Model(irr_python.func, independent_vars=['x'], param_names=variables)
    gmod.set_param_hint('alpha', value=guess[0], min=bounds[0][0], max=bounds[0][1])
    gmod.set_param_hint('beta',  value=guess[1], min=bounds[1][0], max=bounds[1][1])
    gmod.set_param_hint('g_dsa', value=guess[2], min=bounds[2][0], max=bounds[2][1])
    gmod.set_param_hint('g_dsr', value=guess[3], min=bounds[3][0], max=bounds[3][1])

    result_lmfit = gmod.fit(y_python, x=x)
    print(result_lmfit.fit_report())
    #plt.plot(x, y_theano)
    #x_new = np.linspace(300, 900,150)
    #irr_symbol.set_wavelengthAOI(x_new)
    #getIrrRatio = irr_symbol.get_compiled()
    #y_new = getIrrRatio(*expected_values)
    plt.plot(x, y_python)
    plt.xlabel(r"Wavelength $\lambda$ [nm]", **hfont)
    plt.ylabel(r"$E_{ds}/E_{d}$", **hfont)
    plt.legend()
    plt.show()


def sky_radiance():
    from get_ssa import get_ssa
    #zenith = 76.3313400556
    zenith = 60.
    AMass = get_atmospheric_path_length(zenith)

    rel_h = 0.9
    pressure = 950
    ssa = get_ssa(rel_h, 5)
    x = np.linspace(350, 750, 1000)  # config
    H_oz = 0.34
    wv = 0.9
    alpha = 1.0
    beta = 0.1
    l_dsr = 0.17
    l_dsa = 0.13

    model = BaseModelPython(zenith, pressure, ssa)
    skyModel = SkyRadiance(model, model)

    ozone = OzoneTransmittance(model)
    oz = ozone.func(x, H_oz)
    oz2 = ozone.func(x, 0.33)
    oz3 = ozone.func(x, 0.34)

    ww = WaterVapourTransmittance(model)
    water = ww.func(0.25, x)
    water2 = ww.func(1.5, x)

    y_true = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, wv=wv, H_oz=H_oz)
    for wv in np.arange(0.2, 1.2, 0.1):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, wv=wv, H_oz=H_oz)
        plt.plot(x, y-y_true, label=wv)
    plt.xlabel(r'Wavelength $\lambda$ [nm]', **hfont)
    plt.ylabel(r'Deviation Sky Radiance $\Delta$L $\left[\frac{mW}{m^2 \cdot nm \cdot sr}\right]$', **hfont)
    legend = plt.legend(prop=fontP, title=r'Water vapor [cm]')
    legend.get_frame().set_alpha(0.2)
    plt.setp(legend.get_title(),fontsize='small', family=FONTSTYLE)
    plt.show()

    #for hoz in np.arange(0.34, 0.5, 0.01):
    y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, wv=wv, H_oz=H_oz)
    plt.plot(x, y)
    plt.xlabel(r'Wavelength $\lambda$ [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\left[\frac{mW}{m^2 \cdot nm \cdot sr}\right]$', **hfont)
    #plt.title("Ozone 0.3-0.7 cm", **hfont)
    plt.show()

    for l in np.arange(0.12, 0.22, 0.01):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l, wv=wv, H_oz=H_oz)
        plt.plot(x, y, label='%s' % l)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    #plt.title("l_dsr 0.005-0.1", **hfont)
    legend = plt.legend(loc=0, ncol=1, prop = fontP,fancybox=True,
                    shadow=False,title=r'$l_{dsr}$',bbox_to_anchor=(1.0, 1.0))
    plt.setp(legend.get_title(),fontsize='medium', family=FONTSTYLE)
    plt.show()

    for l in np.arange(0.07, 0.5, 0.01):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l, l_dsr=l_dsr, wv=wv, H_oz=H_oz)
        plt.plot(x, y, label='%s' % l)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    #plt.title("l_dsa 0.005-0.1", **hfont)
    legend = plt.legend(loc=0, ncol=1, prop = fontP,fancybox=True,
                    shadow=False,title=r'$l_{dsa}$', bbox_to_anchor=(1.0, 1.0))
    plt.setp(legend.get_title(),fontsize='medium', family=FONTSTYLE)
    plt.show()

    #for a in np.arange(1.4, 18, 0.01):
    #    y = skyModel.func(x=x, alpha=a, beta=beta, l_dsa=l, l_dsr=l_dsr, wv=wv, H_oz=H_oz)
    #    plt.plot(x, y)
    #plt.xlabel('Wavelength [nm]', **hfont)
    #plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    #plt.title("alpha 1.4-1.8", **hfont)
    #plt.show()

    #for b in np.arange(0.02, 0.3, 0.01):
    #    y = skyModel.func(x=x, alpha=alpha, beta=b, l_dsa=l, l_dsr=l_dsr, wv=wv, H_oz=H_oz)
    #    plt.plot(x, y)
    #plt.xlabel('Wavelength [nm]', **hfont)
    #plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    #plt.title("beta 0.02-0.1", **hfont)
    #plt.show()

    plt.plot(x, water, label="WV 0.25cm")
    plt.plot(x, water2, label="WV 1.5cm")
    plt.plot(x,oz, label="Hoz 0.3cm")
    plt.plot(x,oz2, label="Hoz 0.33cm")
    plt.plot(x,oz3, label="Hoz 0.34cm")
    plt.legend(loc='best', prop=fontP)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel('Transmittance', **hfont)
    plt.show()

    for noise in np.arange(20):
        simulation = skyModel.func(x=x, alpha=alpha, beta=0.06, l_dsa=l, l_dsr=l_dsr, wv=wv, H_oz=H_oz)
        plt.plot(x, simulation)
    plt.show()


def l_sky_ratio():
    from get_ssa import get_ssa
    #zenith = 76.3313400556
    zenith = 60.
    AMass = get_atmospheric_path_length(zenith)

    rel_h = 0.9
    pressure = 950
    ssa = get_ssa(rel_h, 5)
    x = np.linspace(350, 750, 1000)  # config
    H_oz = 0.3
    wv = 1.2
    alpha = 1.8
    beta = 0.1
    l_dsr = 0.17
    l_dsa = 0.1
    g_dsr = 0.9
    g_dsa = 0.9


    model = BaseModelPython(zenith, pressure, ssa)
    skyModel = LSkyRatio(model, model)

    for l in np.arange(0.14, 0.22, 0.001):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l, g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("l_dsr 0.14-0.22", **hfont)
    plt.show()

    for l_ in np.arange(0.07, 0.13, 0.001):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_, l_dsr=l_dsr, g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("l_dsa 0.07-0.13", **hfont)
    plt.show()

    for g in np.arange(0.5, 0.99, 0.01):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, g_dsr=g, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("g_dsr 0.5-0.99", **hfont)
    plt.show()

    for g in np.arange(0.5, 0.99, 0.01):
        y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr,g_dsr=g_dsr, g_dsa=g)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("g_dsa 0.5-0.99", **hfont)
    plt.show()

    for a in np.arange(1.4, 2.5, 0.01):
        y = skyModel.func(x=x, alpha=a, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr,g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.title("alpha 1.4-2.5", **hfont)
    plt.show()

    for b in np.arange(0.02, 0.13, 0.001):
        y = skyModel.func(x=x, alpha=alpha, beta=b, l_dsa=l_dsa, l_dsr=l_dsr,g_dsr=g_dsr, g_dsa=g_dsa)
        plt.plot(x, y, label="%s" % b)
    plt.xlabel('Wavelength [nm]', **hfont)
    plt.ylabel(r'Sky Radiance $\frac{mW}{m^2 \cdot nm}$', **hfont)
    plt.legend()
    plt.title("beta 0.02-0.13", **hfont)
    plt.show()

    simulation = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, g_dsr=g_dsr, g_dsa=g_dsa)# + np.random.normal(0, 0.0005, len(x))
    plt.plot(x, simulation)
    plt.xlabel(r'Wavelength $\lambda$ [nm]', **hfont)
    plt.ylabel(r'$L_{sky}/E_d$ $\left[ sr^{-1} \right]$', **hfont)
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
    model = BaseModelPython(zenith, pressure, ssa)
    skyModel = LSkyRatio(model, model)
    y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, g_dsr=g_dsr, g_dsa=g_dsa)

    symmodel = LSkyRatioSym(zenith, pressure, ssa, x, ['alpha', 'beta','g_dsr', 'g_dsa', 'l_dsr', 'l_dsa'])
    func = symmodel.get_compiled()
    ysym = func(alpha, beta, g_dsr, g_dsa, l_dsr, l_dsa)

    plt.plot(x, ysym, label='sym')
    plt.plot(x, y, label='python')
    plt.legend()
    plt.show()


def fit_skyRadiance():
    from scipy.optimize import minimize, least_squares
    from get_ssa import get_ssa
    zenith = 53.1836240528
    AMass = 1.66450160404
    rel_h = 0.665
    pressure = 950
    AM = 5
    ssa = get_ssa(rel_h, AM)
    x = np.linspace(350, 700, 1000)  # config
    H_oz = 0.34
    wv = 1.2
    alpha = 1.8
    beta = 0.06
    l_dsr = 0.05
    l_dsa = 0.05
    guess = [alpha+0.2, beta+0.02, l_dsr+0.01,  l_dsa+0.01]  # config
    bounds = [(-0.25, 4.0), (0.01, 0.3), (-0.05, 0.1), (-0.05, 0.3)]  # config

    symmodel = SkyRadianceSym(zenith, pressure, ssa, x, ['alpha', 'beta', 'l_dsr', 'l_dsa', 'H_oz', 'wv'])
    func = symmodel.get_compiled()
    simulation = func(alpha, beta, l_dsr, l_dsa, H_oz, wv) #+ np.random.normal(0, 0.1, len(x))
    plt.plot(x, simulation)
    plt.show()
    symmodel.setVariable('H_oz', 0.3)
    symmodel.setVariable('wv', 0.1)
    print(symmodel.get_Symbols())
    res = Residuum(symmodel)
    residuals = FitWrapper(res.getResiduum())
    result = minimize(residuals, guess, args=(simulation), jac=False, method='L-BFGS-B', bounds=bounds)
    print(result.x)
    print(result.success)
    plt.plot(x, simulation, label='sym')
    func = symmodel.get_compiled()
    fitted = func(*result.x)
    plt.plot(x, fitted)
    plt.show()
    for noise in np.arange(20):
        simulation = func(alpha, beta, l_dsr, l_dsa) + np.random.normal(0, 0.04, len(x))
        plt.plot(x, simulation)
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

    symmodel = SkyRadianceSym(zenith, pressure, ssa, x, ['alpha', 'beta', 'l_dsr', 'l_dsa', 'H_oz', 'wv'])
    func = symmodel.get_compiled()
    ysym = func(alpha, beta, l_dsr, l_dsa, H_oz, wv)
    plt.plot(x, ysym, label='sym')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_main()
    sky_radiance()
    l_sky_ratio()
    #compare_sym_python()
    #coverty_variability()
    #fit_skyRadiance()
