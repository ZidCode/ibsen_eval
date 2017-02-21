import theano
import numpy as np
from math import exp, log
from theano import tensor as T
from BaseModels import BaseModelSym


# Python: Composition, Sym: Inheritance
exp_v = np.vectorize(exp)


class WaterVapourTransmittance:
    def __init__(self, base_model):
        self.bm = base_model

    def func(self, WV, x):
        return exp_v(self.bm.tau_wv(WV, x))


class OzoneTransmittance:
    def __init__(self, base_model):
        self.bm = base_model

    def func(self, x, H_oz):
        return exp_v(self.bm.tau_oz(x, H_oz))


class IrradianceRatio:

    def __init__(self, base_model):
        self.bm = base_model

    def func(self, x, alpha, beta, g_dsr=1, g_dsa=1, g_dd=0):
        term = self.bm.ratio_sky_radiance(x, alpha, beta, g_dsa=g_dsa, g_dsr=g_dsr, g_dd=g_dd)
        nom = self.bm.ratio_sky_radiance(x, alpha, beta)
        return term / nom


class LSkyRatio:

    def __init__(self, base_model):
        self.bm = base_model

    def func(self, x, alpha, beta, l_dsa, l_dsr, g_dsr=0.4, g_dsa=0.4, l_dd=0, g_dd=1):
        term = self.bm.ratio_sky_radiance(x, alpha, beta, g_dsr=l_dsr, g_dsa=l_dsa, g_dd=l_dd)
        norm = self.bm.ratio_sky_radiance(x, alpha, beta, g_dsr=g_dsr, g_dsa=g_dsa, g_dd=g_dd)
        return term / norm


class SkyRadiance:

    def __init__(self, base_model):
        self.bm = base_model


    def func(self, x, alpha, beta, l_dsr, l_dsa, H_oz, wv):
        E0 = np.interp(x, self.bm.wasi['e0']['wave'], self.bm.wasi['e0']['values'])
        non_aerosol_term = E0 * np.cos(self.bm.zenith_rad) * exp_v(self.bm.tau_oz(x, H_oz)) * exp_v(self.bm.tau_o2(x)) * exp_v(self.bm.tau_wv(wv, x))
        aerosol_term  = l_dsr * (1 - exp_v(0.95 * self.bm.tau_r(x))) * 0.5 + \
                        l_dsa * exp_v(1.5 * self.bm.tau_r(x)) * (1 - exp_v(self.bm.tau_as(x, alpha, beta))) * self.bm.forward_scat(alpha)
        return non_aerosol_term * aerosol_term


class IrradianceRatioSym(BaseModelSym):

    def __init__(self, zenith, AM, pressure, ssa, wave, variables=['alpha', 'beta', 'g_dsa', 'g_dsr']):
        BaseModelSym.__init__(self, zenith, AM, pressure, ssa, wave, variables)

    def _ratio_E_ds(self):
        _E_ds_ratio = self.symbols['g_dsr'] * (1 - T.exp(0.95 * self._tau_r())) * 0.5 + \
                      self.symbols['g_dsa'] * T.exp(1.5 * self._tau_r()) * (1 - T.exp(self._tau_as())) * self._forward_scat()
        return _E_ds_ratio

    def _ratio_E_d(self):
        _E_d_ratio = (1 - T.exp(0.95 * self._tau_r())) * 0.5 + \
                     T.exp(1.5 * self._tau_r()) * (1 - T.exp(self._tau_as())) * self._forward_scat() +  \
                     T.exp(self._tau_as() + self._tau_r())
        return _E_d_ratio

    def func(self):
        return self._ratio_E_ds() / self._ratio_E_d()

    def get_compiled(self):
        symbols = self.get_Symbols()
        call_model = theano.function(symbols, self.func())
        return call_model


class SkyRadianceSym(BaseModelSym):

    def __init__(self, zenith, AM, pressure, ssa, wave, variables=['alpha', 'beta', 'l_dsr', 'l_dsa', 'H_oz', 'wv']):
        BaseModelSym.__init__(self, zenith, AM, pressure, ssa, wave, variables)


    def func(self):
        non_aerosol = self.solar * T.cos(self.zenith_rad) * T.exp(self._tau_oz()) * T.exp(self._tau_wv()) * T.exp(self._tau_o2())
        aerosol = self.symbols['l_dsr'] * (1 - T.exp(0.95 * self._tau_r())) * 0.5 + \
                      self.symbols['l_dsa'] * T.exp(1.5 * self._tau_r()) * (1 - T.exp(self._tau_as())) * self._forward_scat()
        return non_aerosol * aerosol

    def get_compiled(self):
        symbols = self.get_Symbols()
        call_model = theano.function(symbols, self.func())
        return call_model

