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
        leastsq
        tnc
    """
    def __init__(self, method_dict={'method':'TNC', 'jac_flag': True}):
        self.method = method_dict['method']
        self.jac_flag = method_dict['jac_flag']
        self.result = None

    def _minimize(self, thcallable, start, y, bounds, jacobian=False):
        self.result = minimize(thcallable, start, args=(y), jac=jacobian, method=self.method, bounds=bounds)
        return self.result

    def _least_squares(self, thecallable, start, y, bounds, jacobian=False):
        bounds = tuple(map(list, zip(*bounds)))  # [(min,max),(min1,max1)..] -> ([min,min1,..], [max,max1..])
        if jacobian:
            self.result = least_squares(thecallable, start, jac=jacobian, args=(y,), bounds=bounds)
        else:
            self.result = least_squares(thecallable, start, args=(y,), bounds=bounds)
        return self.result
