from lmfit import Model
from scipy.optimize import minimize, least_squares
from Model import IrradianceModel_python, IrradianceModel_sym
from Residuum import Residuum


class FitWrapper:

    def __init__(self, model):
        self.symbolic_model = model

    def __call__(self, params, y):
        res = self.symbolic_model(y, *params)
        return res


class FitModelFactory:

    def __init__(self, wp, config, wavelengths, logger):

        if config['package'] == 'lmfit':
            self.model = IrradianceModel_python(wp.sun_zenith, wp.atmos_path, wp.pressure, wp.ssa)
        else:
            self.model = IrradianceModel_sym(wp.sun_zenith, wp.atmos_path, wp.pressure, wp.ssa, wavelengths, config['params'])

    def get_fitmodel(self):
        return self.model


class FitMethodFactory:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def get_method(self):
        if self.config['package'] =='lmfit':
            self.logger.info("Using lmfit package")
            return LMFit
        elif self.config['package'] == 'least_squares':
            self.logger.info("Using least_squares package")
            return LeastSquaresFit
        elif self.config['package'] == 'minimize':
            self.logger.info("Using minimize package")
            return Minimize


class LMFit:

    def __init__(self, model, config, param_dict, logger):
        self.config = config
        self.param_dict = param_dict
        self.logger = logger
        self.callable = model.irradiance_ratio

    def fit(self):
        self.logger.info('Method %s' % self.config['method'])
        gmod = Model(self.callable, independent_vars=['x'], param_names=self.config['params'], method=self.config['method'])
        self._set_params(self.config['params'], self.config['initial_values'], self.config['limits'], gmod)
        self.result = gmod.fit(self.param_dict['spectra_range'], x=self.param_dict['wave_range'])
        for key in self.result.params.keys():
            self.param_dict[key] = dict()
            self.param_dict[key]['stderr'] = self.result.params[key].stderr
            self.param_dict[key]['value'] = self.result.params[key].value
        return self.result, self.param_dict

    def _set_params(self, params, initial_values, limits, fit_model):
        for param, ini, limits in zip(params, initial_values, limits):
            self.logger.info("Setting for %s: initial: %s and bound %s" % (param, ini, limits))
            fit_model.set_param_hint(param, value=ini, min=limits[0], max=limits[1])


class Minimize:

    def __init__(self, model, config, param_dict, logger):
        self.model = model
        self.res = Residuum(model, 'ratio')
        self.callable = FitWrapper(self.res.getResiduum())
        self.symbols = model.get_Symbols()
        self.param_dict = param_dict
        self.config = config
        self.logger = logger
        if config['jac_flag']:
            logger.info("Using jacobian")
            self.jacobian = FitWrapper(self.res.getDerivative())
        else:
            assert config['jac_flag'] == False
            self.jacobian = False

    def fit(self):
        self.logger.info('Method %s' % self.config['method'])
        self.result = minimize(self.callable, self.config['initial_values'], args=(self.param_dict['spectra_range']), jac=self.jacobian,
                               method=self.config['method'], bounds=self.config['limits'])
        for idx, symbol in enumerate(self.symbols):
            self.param_dict[symbol] = dict()
            self.param_dict[symbol]['stderr'] = None
            self.param_dict[symbol]['value'] = self.result.x[idx]
        self._calc_fitted_spectra()
        self._calc_residuals()
        return Result(self.result, self.fitted_spectra, self.residuals, self.param_dict['wave_range']), self.param_dict

    def _calc_fitted_spectra(self):
        f = self.model.getcompiledModel('ratio')
        self.fitted_spectra = f(*self.result.x)

    def _calc_residuals(self):
        #TODO
        self.residuals = self.param_dict['spectra_range'] - self.fitted_spectra


class LeastSquaresFit:

    def __init__(self, model, config, param_dict, logger):
        self.config = config
        self.model = model
        self.res = Residuum(model, 'ratio')
        self.callable = FitWrapper(self.res.getResiduals())
        self.symbols = self.model.get_Symbols()
        self.param_dict = param_dict
        self.logger = logger

    def fit(self):
        self.logger.info('Method %s' % self.config['method'])
        bounds = tuple(map(list, zip(*self.config['limits'])))  # [(min,max),(min1,max1)..] -> ([min,min1,..], [max,max1..])
        self.result = least_squares(self.callable, self.config['initial_values'], args=(self.param_dict['spectra_range'],), bounds=bounds)
        for idx, symbol in enumerate(self.symbols):
            self.param_dict[symbol] = dict()
            self.param_dict[symbol]['stderr'] = None
            self.param_dict[symbol]['value'] = self.result.x[idx]
        return Result(self.result), self.param_dict


class Result:
    def __init__(self, result, fitted_spectra, residuals, wavelength):
        #public
        self.result = result
        self.best_fit = fitted_spectra
        self.residuals = residuals
        self.wavelength = wavelength

    def fit_report(self):
        print(self.result)

    def plot_residuals(self, ax):
        ax1 = ax.plot(self.wavelength, self.residuals)
        return ax1


