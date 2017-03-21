import theano
import numpy as np
from math import exp, log
from theano import tensor as T
from scipy.constants import atmosphere
from wasi_reader import get_wasi_parameters, get_wasi
from atmospheric_mass import get_ozone_path_length, get_atmospheric_path_length


exp_v = np.vectorize(exp)


class BaseModelPython:

    def __init__(self, zenith, pressure, ssa):
        self.AM = get_atmospheric_path_length(zenith)
        self.ssa = ssa
        self.pressure = pressure
        self.p_0 = atmosphere / 100.0
        self.zenith_rad = np.radians(zenith)
        self.ray = 0.00877
        self.ray_expo = -4.05
        self.lambda_reference = 550  # [nm] Gege, 1000 nm Greg and Carder + Bringfried
        self.wasi = get_wasi_parameters()  # dict
        self.AM_ozone = get_ozone_path_length(zenith)  # zenith = degrees

    def forward_scat(self, alpha):
        """
        Reference: Greg and Carder
        empirical values
        """
        if alpha > 1.2:
            cos_theta = 0.65
        else:
            cos_theta = -0.1417 * alpha + 0.82
        B_3 = log(1 - cos_theta)
        B_2 = B_3 * (0.0783 + B_3 * (-0.3824 - 0.5874 * B_3))
        B_1 = B_3 * (1.459 + B_3 * (0.1595 + 0.4129 * B_3))
        F_a = 1 - 0.5 * exp_v((B_1 + B_2 * np.cos(self.zenith_rad)) * np.cos(self.zenith_rad))
        return F_a

    def tau_r(self, x):
        y = - ( self.AM * self.pressure/self.p_0) / (115.640 * (x/1000) ** 4 - 1.335 * (x/1000)**2)
        return y

    def tau_as(self, x, alpha, beta):
        return - self.ssa * self.AM * beta * (x / self.lambda_reference)  ** (-alpha)

    def tau_aa(self, x, alpha, beta):
        return - (1 - self.ssa) * self.AM * beta * (x / self.lambda_reference)  ** (-alpha)

    def tau_oz(self, x, H_oz):
        "Ozone Transmittance"
        oz = np.interp(x, self.wasi['o3']['wave'], self.wasi['o3']['values'])
        return - oz * H_oz * self.AM_ozone

    def tau_o2(self, x):
        o2 = np.interp(x, self.wasi['o2']['wave'], self.wasi['o2']['values'])
        term = -1.41 * o2 * ( self.AM * self.pressure/self.p_0)
        norm = (1 + 118.3 * o2 * ( self.AM * self.pressure/self.p_0)) ** 0.45
        return term / norm

    def tau_wv(self, WV, x):
        wv = np.interp(x, self.wasi['wv']['wave'], self.wasi['wv']['values'])
        term = -0.2385 * wv * WV * self.AM
        norm = (1 + 20.07 * wv * WV * self.AM) ** 0.45
        return term / norm

    def ratio_sky_radiance(self, x, alpha, beta, g_dsr=1, g_dsa=1, g_dd=1):
        term = g_dsr * (1 - exp_v(0.95 * self.tau_r(x))) * 0.5 + \
               g_dsa * exp_v(1.5 * self.tau_r(x)) * (1 - exp_v(self.tau_as(x, alpha, beta))) * self.forward_scat(alpha) +  \
               g_dd * exp_v(self.tau_as(x, alpha, beta) + self.tau_r(x))
        return term


class BaseModelSym:
    """ Only symbolic representation"""
    def __init__(self, zenith, pressure, ssa, wave, variables=['alpha', 'beta', 'g_dsa', 'g_dsr']):
        #private
        self.variables = variables
        self.wavelength = theano.shared(wave, 'wavelength')
        self.zenith_rad = theano.shared(np.radians(zenith), 'zenith_rad')
        self.AM = theano.shared(get_atmospheric_path_length(zenith), 'AM')
        self.pressure = theano.shared(pressure, 'pressure')
        self.ssa = theano.shared(ssa, 'ssa')
        ozone, oxygen, water, solar = get_wasi(wave)
        self.o3 = theano.shared(ozone, 'o3')
        self.o2 = theano.shared(oxygen, 'o2')
        self.wv = theano.shared(water, 'wv')
        self.solar = theano.shared(solar, 'e0')
        self.AM_ozone = theano.shared(get_ozone_path_length(zenith), 'AM_ozone')
        self._lambda_reference = T.constant(550.)
        self.symbols = {'alpha': T.scalar('alpha'), 'beta': T.scalar('beta'), 'g_dsa': T.scalar('g_dsa'), 'g_dsr': T.scalar('g_dsr'),
                        'wv': T.scalar('wv'), 'H_oz': T.scalar('H_oz'), 'l_dsa': T.scalar('l_dsa'), 'l_dsr': T.scalar('l_dsr')}

    def set_wavelengthAOI(self, wave):
        self.wavelength = theano.shared(wave, 'wavelength')

    def get_Symbols(self):
        return [self.getVariable(name) for name in self.variables if type(self.getVariable(name)) == T.TensorVariable]

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

    def _tau_aa(self):
        return - (1 - self.ssa) * self.AM * self.symbols['beta'] * (self.wavelength / self.lambda_reference)  ** (-self.symbols['alpha'])

    def _tau_oz(self):
        "Ozone Transmittance"
        return - self.o3 * self.symbols['H_oz'] * self.AM_ozone

    def _tau_o2(self):
        p_0 = T.constant(atmosphere / 100.)
        term = -1.41 * self.o2 * ( self.AM * self.pressure / p_0)
        norm = (1 + 118.3 * self.o2 * ( self.AM * self.pressure / p_0)) ** 0.45
        return term / norm

    def _tau_wv(self):
        term = -0.2385 * self.wv * self.symbols['wv'] * self.AM
        norm = (1 + 20.07 * self.wv * self.symbols['wv'] * self.AM) ** 0.45
        return term / norm
