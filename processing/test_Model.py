import theano
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from BaseModels import BaseModelPython
from Model import SkyRadiance, WaterVapourTransmittance, OzoneTransmittance, SkyRadianceSym
from FitModel import FitWrapper
from Residuum import Residuum


def gaussian(x, a):
    return a[0] * np.exp(-0.5*(x-a[1])**2/a[2]**2)


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
    expected_values = [2.5, 0.06, 0.6, 0.5]

    print('Expected: %s' % expected_values)
    guess = [1.0, 0.01, 0.5, 0.8]  # config
    bounds = [(-0.2, 4), (0., 3), (0., 2.), (0., 2.)]  # config

    # Theano
    irr_symbol = IrradianceModel_sym(x, zenith, AMass, pressure, ssa, variables)
    getIrrRatio = irr_symbol.getcompiledModel('ratio')
    y_theano = getIrrRatio(*expected_values)
    res = Residuum(irr_symbol, 'ratio')
    residuum = FitWrapper(res.getResiduum())
    residuals = FitWrapper(res.getResiduals())
    derivative = FitWrapper(res.getDerivative())


    # Python
    IrradianceObject = IrradianceModel_python(AMass, rel_h, ssa, zenith, pressure)
    y_python = IrradianceObject.irradiance_ratio(x, 2.5, 0.06,0.0, 0.6, 0.5)

    gmod = Model(IrradianceObject.irradiance_ratio, independent_vars=['x'], param_names=variables)
    gmod.set_param_hint('alpha', value=guess[0], min=bounds[0][0], max=bounds[0][1])
    gmod.set_param_hint('beta',  value=guess[1], min=bounds[1][0], max=bounds[1][1])
    gmod.set_param_hint('g_dsa', value=guess[2], min=bounds[2][0], max=bounds[2][1])
    gmod.set_param_hint('g_dsr', value=guess[3], min=bounds[3][0], max=bounds[3][1])

    result_lmfit = gmod.fit(y_python, x=x)
    print(result_lmfit.fit_report())

    plt.plot(x, y_theano)
    x_new = np.linspace(300, 900,150)
    irr_symbol.set_wavelengthAOI(x_new)
    getIrrRatio = irr_symbol.getcompiledModel('ratio')
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
    x = np.linspace(350, 800, 1000)  # config
    H_oz = 0.3
    wv = 0.25
    alpha = 1.6
    beta = 0.06
    l_dsr = 0.02
    l_dsa = 0.02

    model = BaseModelPython(zenith, AM, pressure, ssa)
    skyModel = SkyRadiance(model)
    waterv = OzoneTransmittance(model)

    y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, wv=wv, H_oz=H_oz)
    y2 = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr,wv=0.0, H_oz=H_oz)
    y3 = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, wv=0.5, H_oz=H_oz)

    wv = waterv.func(x, H_oz)
    wv2 = waterv.func(x, 0.33)
    wv3 = waterv.func(x, 0.34)
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.ylabel('sky radiance')
    plt.show()
    plt.plot(x,wv)
    plt.plot(x,wv2)
    plt.plot(x,wv3)
    plt.show()

    ww = WaterVapourTransmittance(model)
    water = ww.func(0.25, x)
    water2 = ww.func(0.3, x)
    plt.plot(x, water)
    plt.plot(x, water2)
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
    H_oz = 0.3
    wv = 0.25
    alpha = 1.6
    beta = 0.06
    l_dsr = 0.02
    l_dsa = 0.02
    model = BaseModelPython(zenith, AM, pressure, ssa)
    skyModel = SkyRadiance(model)
    y = skyModel.func(x=x, alpha=alpha, beta=beta, l_dsa=l_dsa, l_dsr=l_dsr, wv=wv, H_oz=H_oz)

    symmodel = SkyRadianceSym(zenith, AM, pressure, ssa, x, ['alpha', 'beta', 'l_dsr', 'l_dsa', 'H_oz', 'wv'])
    func = symmodel.get_compiled()
    ysym = func(alpha, beta, l_dsr, l_dsa, H_oz, wv)

    plt.plot(x, ysym, label='sym')
    plt.plot(x, y, label='python')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # example_gaussian()
    # test_main()
    #sky_radiance()
    compare_sym_python()