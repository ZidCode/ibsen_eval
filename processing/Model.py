import theano
import numpy as np
from theano import tensor as T
from scipy.constants import atmosphere


class IrradianceModel:
    """ Only symbolic representation"""
    def __init__(self, wave, zenith=0, AM=0, pressure=0, ssa=0, variables=['alpha', 'beta', 'g_dsa', 'g_dsr']):
        #private
        self.variables = variables
        self.wavelength = theano.shared(wave, 'wavelength')
        self.zenith_rad = theano.shared(np.radians(zenith), 'zenith_rad')
        self.AM = theano.shared(AM, 'AM')
        self.pressure = theano.shared(pressure, 'pressure')
        self.ssa = theano.shared(ssa, 'ssa')

        self._lambda_reference = T.constant(1000.)
        #public
        self.model_dict = {'ratio': self._irradiance_ratio, 'nadir': self._radiance_nadir, 'E_d': self._irradiance}
        self.symbols = {'alpha': T.scalar('alpha'), 'beta': T.scalar('beta'), 'g_dsa': T.scalar('g_dsa'), 'g_dsr': T.scalar('g_dsr')}

    def get_Symbols(self):
        return [self.getVariable(name) for name in self.variables if type(self.getVariable(name)) == T.TensorVariable]

    def getModel(self, name):
        return self.model_dict[name]()

    def getcompiledModel(self, name):
        symbolic = self.getModel(name)
        symbols = self.get_Symbols()
        call_model = theano.function(symbols, symbolic)
        return call_model

    def getVariable(self, name):
        return self.symbols[name]

    def setVariable(self, name, value):
        self.symbols[name] = value

    def resettoTensorVariable(self, name):
        self.setVariable(name, T.scalar(name))

    def _forward_scat(self):
        alpha_border = 1.2  # Greg & Carder
        cos_theta = T.switch(T.gt(self.symbols['alpha'], alpha_border), 0.65, -0.1417 * self.symbols['alpha'] + 0.82)
        B3 = T.log(1 - cos_theta)
        B2 = B3 * (0.0783 + B3 * (-0.3824 - 0.5874 * B3))
        B1 = B3 * (1.459 + B3 * (0.1595 + 0.4129 * B3))
        return 1 - 0.5 * T.exp((B1 + B2 * T.cos(self.zenith_rad)) * T.cos(self.zenith_rad))

    def _tau_r(self):
        "Rayleigh optical depth"
        p_0 = T.constant(atmosphere / 100.)
        tau_r = - (self.AM * self.pressure / p_0) / (115.640 * (self.wavelength / 1000) ** 4 - 1.335 * (self.wavelength / 1000) ** 2)
        return tau_r

    def _tau_as(self):
        "Aerosol scattering optical depth"
        tau_as = - self.ssa * self.AM * self.symbols['beta'] * ( self.wavelength / self._lambda_reference) ** (-self.symbols['alpha'])
        return tau_as

    def _ratio_E_ds(self):
        _E_ds_ratio = self.symbols['g_dsr'] * (1 - T.exp(0.95 * self._tau_r())) * 0.5 + \
                      self.symbols['g_dsa'] * T.exp(1.5 * self._tau_r()) * (1 - T.exp(self._tau_as())) * self._forward_scat()
        return _E_ds_ratio

    #Duplicated (bad)
    def _ratio_E_d(self):
        _E_d_ratio = (1 - T.exp(0.95 * self._tau_r())) * 0.5 + \
                     T.exp(1.5 * self._tau_r()) * (1 - T.exp(self._tau_as())) * self._forward_scat() +  \
                     T.exp(self._tau_as() + self._tau_r())
        return _E_d_ratio

    def _irradiance_ratio(self):
        return self._ratio_E_ds() / self._ratio_E_d()

    def _radiance_nadir(self):
        pass

    def _irradiance(self):
        pass


class TmpModel:
    def __init__(self, wave, variables):
        self.x = theano.shared(wave, borrow=True)
        self.variables = variables

        self.a = T.scalar('a')
        self.b = T.scalar('b')
        self.c = T.scalar('c')
        self.symbols = {'a': self.a, 'b': self.b, 'c': self.c}
        self.model_dict = {'gauss': self._symbolic_gaussian}

    def get_Symbols(self):
        return [self.symbols[name] for name in self.variables]

    def _symbolic_gaussian(self):
        y = self.a * T.exp(-0.5*(self.x-self.b) ** 2 / self.c**2)
        return y

    def getModel(self, name):
        return self.model_dict[name]
