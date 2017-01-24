from lmfit import Model
from scipy.optimize import minimize, least_squares


class FitWrapper:

    def __init__(self, model):
        self.symbolic_model = model

    def __call__(self, params, y):
        res = self.symbolic_model(y, *params)
        return res


class FitModel:
    """
    lmfit/scipy
        minimize
        curve_fit
        least_squares
    """
    def __init__(self, method='TNC', fit_model=None):
        self.method = method
        self.result = None
        self.fit_model = fit_model

    def minimize(self, thcallable, start, y, bounds, jacobian=False):
        self.result = minimize(thcallable, start, args=(y), jac=jacobian, method=self.method, bounds=bounds)
        return self.result

    def least_squares(self, thecallable, start, y, bounds, jacobian=False):
        bounds = tuple(map(list, zip(*bounds)))  # [(min,max),(min1,max1)..] -> ([min,min1,..], [max,max1..])
        if jacobian:
            self.result = least_squares(thecallable, start, jac=jacobian, args=(y,), bounds=bounds)
        else:
            self.result = least_squares(thecallable, start, args=(y,), bounds=bounds)
        return self.result

    def LMfit(self, wavelength, thecallable, params, start, y, bounds, jacobial=False):
        self.fit_model = Model(thecallable, independent_vars=['x'], param_names=params)
        self._set_params(params, start, bounds)
        self.result = self.fit_model.fit(y, x=wavelength)
        return self.result

    def _set_params(self, params, initial_values, limits):
        for param, ini, limits in zip(params, initial_values, limits):
            self.fit_model.set_param_hint(param, value=ini, min=limits[0], max=limits[1])
