import theano
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from Model import IrradianceModel_sym, IrradianceModel_python, GaussModel
from FitModel import FitWrapper, FitModel
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
    x = np.linspace(200, 800, 100)  # config
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
    Fit = FitModel()
    result = Fit._minimize(residuum, guess, y_theano, bounds, jacobian=derivative)
    print("Got %s" % result.x)
    resultls = Fit._least_squares(residuals, guess, y_theano, bounds)
    print("Got %s" % resultls.x)


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


def example_gaussian():
    #Config output
    x = np.linspace(-20, 20, 100)
    a = 3.4
    b = 2.4
    c = 5
    bounds = [(3, 4), (2, 3), (4.2, 5.2)]
    guess = [a,b,c]
    variables = ['a', 'b', 'c']
    y = gaussian(x, guess) + np.random.normal(0, 0.1, len(x))

    #Auswertung
    model= GaussModel(x, variables)
    res = Residuum(model, 'gauss')
    function_to_fit = FitWrapper(res.getResiduum())
    test = FitModel(function_to_fit)
    result = test.fit(guess, y, bounds)

    y_fitted = gaussian(x, result.x)

    plt.plot(x,y, 'b')
    plt.plot(x, y_fitted, 'r')


if __name__ == "__main__":
    # example_gaussian()
    test_main()

