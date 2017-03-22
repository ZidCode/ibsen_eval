from lmfit import Model
from scipy.optimize import minimize, least_squares
from BaseModels import BaseModelPython, BaseModelSym
from Model import IrradianceRatio, LSkyRatio, SkyRadiance, IrradianceRatioSym, SkyRadianceSym
from Residuum import Residuum


class FitWrapper:

    def __init__(self, model):
        self.symbolic_model = model

    def __call__(self, params, y):
        res = self.symbolic_model(y, *params)
        return res


class FitModelFactory:

    def __init__(self, wp, config, wavelengths, logger):
        #'ratio', 'l_sky_ratio', 'l_sky_nadir'
        python_map = {'ratio': IrradianceRatio, 'l_sky_ratio': LSkyRatio, 'l_sky_nadir': SkyRadiance}
        sym_map = {'ratio': IrradianceRatioSym, 'l_sky_nadir': SkyRadianceSym}

        if config['package'] == 'lmfit':
            bm = BaseModelPython(wp.sun_zenith_tar, wp.pressure, wp.ssa)
            bmref = BaseModelPython(wp.sun_zenith_ref, wp.pressure, wp.ssa)
            self.model = python_map[config['model']](bm, bmref)
        else:
            self.model = sym_map[config['model']](wp.sun_zenith_tar, wp.pressure, wp.ssa, wavelengths, config['params'])

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
        self.callable = model.func

    def fit(self):
        self.logger.info('Method %s' % self.config['method'])
        self.logger.info('Parameter  names %s' % self.config['params'])

        gmod = Model(self.callable, independent_vars=self.config['independent'].keys(), param_names=self.config['params'], method=self.config['method'])
        self._set_params(self.config['params'], self.config['initial_values'], self.config['limits'], gmod)
        self.config['independent']['x'] = self.param_dict['wave_range']
        helper_param = {key:val for key, val in self.config['independent'].items() if key != 'x'}
        self.logger.debug("Setting %s parameters fix" % helper_param)
        self.result = gmod.fit(self.param_dict['spectra_range'], weights=self.param_dict['weights'],  **self.config['independent'])
        self.param_dict['variables'] = dict()
        for key in self.result.params.keys():
            self.param_dict['variables'][key] = dict()
            self.param_dict['variables'][key]['stderr'] = self.result.params[key].stderr
            self.param_dict['variables'][key]['value'] = self.result.params[key].value
        return self.result, self.param_dict

    def _set_params(self, params, initial_values, limits, fit_model):
        for param, ini, limits in zip(params, initial_values, limits):
            self.logger.info("Setting for %s: initial: %s and bound %s" % (param, ini, limits))
            fit_model.set_param_hint(param, value=ini, min=limits[0], max=limits[1])


class Minimize:

    def __init__(self, model, config, param_dict, logger):
        self.model = model
        self.param_dict = param_dict
        self.config = config
        del self.config['independent']['x']
        for key, value in self.config['independent'].items():
            logger.debug("Set %s for %s " % (value, key))
            self.model.setVariable(key, value)
        self.res = Residuum(model)
        self.callable = FitWrapper(self.res.getResiduum())
        self.symbols = model.get_Symbols()
        self.logger = logger
        if config['jac_flag']:
            logger.info("Using jacobian")
            self.jacobian = FitWrapper(self.res.getDerivative())
        else:
            assert config['jac_flag'] == False
            self.jacobian = False

    def fit(self):
        self.logger.info("Symbols: %s" % self.symbols)
        self.logger.info('Method %s' % self.config['method'])
        self.result = minimize(self.callable, self.config['initial_values'], args=(self.param_dict['spectra_range']), jac=self.jacobian,
                               method=self.config['method'], bounds=self.config['limits'])
        self.param_dict['variables'] = dict()
        for idx, symbol in enumerate(self.symbols):
            self.param_dict['variables'][symbol] = dict()
            self.param_dict['variables'][symbol]['stderr'] = None
            self.param_dict['variables'][symbol]['value'] = self.result.x[idx]
        self._calc_fitted_spectra()
        self._calc_residuals()
        return Result(self.result, self.fitted_spectra, self.residuals, self.param_dict['wave_range']), self.param_dict

    def _calc_fitted_spectra(self):
        f = self.model.get_compiled()
        self.fitted_spectra = f(*self.result.x)

    def _calc_residuals(self):
        #TODO
        self.residuals = self.param_dict['spectra_range'] - self.fitted_spectra


class LeastSquaresFit:

    def __init__(self, model, config, param_dict, logger):
        self.config = config
        self.model = model
        self.res = Residuum(model)
        self.callable = FitWrapper(self.res.getResiduals())
        self.symbols = model.get_Symbols()
        self.param_dict = param_dict
        self.logger = logger

    def fit(self):
        self.logger.info("Symbols: %s" % self.symbols)
        self.logger.info('Method %s' % self.config['method'])
        bounds = tuple(map(list, zip(*self.config['limits'])))  # [(min,max),(min1,max1)..] -> ([min,min1,..], [max,max1..])
        self.result = least_squares(self.callable, self.config['initial_values'], args=(self.param_dict['spectra_range'],), bounds=bounds)
        self.param_dict['variables'] = dict()
        for idx, symbol in enumerate(self.symbols):
            self.param_dict['variables'][symbol] = dict()
            self.param_dict['variables'][symbol]['stderr'] = None
            self.param_dict['variables'][symbol]['value'] = self.result.x[idx]
        self._calc_fitted_spectra()
        self._calc_residuals()
        return Result(self.result, self.fitted_spectra, self.residuals, self.param_dict['wave_range']), self.param_dict

    def _calc_fitted_spectra(self):
        f = self.model.get_compiled()
        self.fitted_spectra = f(*self.result.x)

    def _calc_residuals(self):
        #TODO
        self.residuals = self.param_dict['spectra_range'] - self.fitted_spectra


class Result:
    def __init__(self, result, fitted_spectra, residuals, wavelength):
        #public
        self.result = result
        self.success = result.success
        self.best_fit = fitted_spectra
        self.residuals = residuals
        self.wavelength = wavelength

    def fit_report(self):
        print(self.result)

    def plot_residuals(self, ax):
        ax1 = ax.plot(self.wavelength, self.residuals)
        return ax1


