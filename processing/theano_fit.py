import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from theano import tensor as T
import theano


class IrradianceModel:

    def __init__(self, wave):
        self.wavelength = wave

    def symbolic_gaussian(self):
        a = T.scalar('a')
        b = T.scalar('b')
        c = T.scalar('c')
        x = theano.shared(self.wavelength, borrow=True)
        reference = T.vector('reference')
        y = a * T.exp(-0.5*(x-b) ** 2 / c**2)
        R = 0.5* T.sum((y-reference)**2)
        dR = T.grad(R, [a, b, c])
        return theano.function([reference, a, b, c], [R] + dR)


def gaussian(x, a):
    return a[0] * np.exp(-0.5*(x-a[1])**2/a[2]**2)


class FitWrapper:

    def __init__(self, model, params):
        self.model = model

    def fit_wrapper(self, params, y):
        f = self.model.symbolic_gaussian()
        res = f(y, *params)
        return res[0], res[1:]

    def fit_wrapper2(self, params, y):
        f = self.model.symbolic_gaussian()
        res = f(y, *params)
        return res[0]


class FitModel:

    def __init__(self, thModel):
        self.thModel = thModel

    def fit(start, y):
        result = minimize(thModel.fit_wrapper, [a,b,c], args=(y), jac=True, method='TNC', bounds = bounds)


if __name__ == "__main__":
    #Config output
    x = np.linspace(-20, 20, 100)
    a = 3.4
    b = 2.4
    c = 5
    bounds = [(3, 4), (2, 3), (4.5, 4.2)]
    params =[a,b,c]
    y = gaussian(x, params) + np.random.normal(0, 0.1, len(x))

    #Auswertung
    model=IrradianceModel(x)
    _try = FitWrapper(model, params)
    result = minimize(_try.fit_wrapper, [a,b,c], args=(y), jac=True, method='TNC', bounds = bounds)
    print(result.x)

    y_fitted = gaussian(x, result.x)
    plt.plot(x,y, 'b')
    plt.plot(x, y_fitted, 'r')
    plt.show()
