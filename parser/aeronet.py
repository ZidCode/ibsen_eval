#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from lmfit import Model
from utils.plotting import plot_turbidity


class AeronetParser:

    def __init__(self):
        self.aeronet_dict = dict()
        self.convert2datetime = lambda d: datetime.strptime(d, '%d:%m:%Y %H:%M:%S')
        self.merge = lambda tup: tup[0] + ' ' + tup[1]
        self.padding = None
        self.timeline = None
        self.column_names = None
        self.logger = logging.getLogger('Aeronet')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s - %(funcName)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.setLevel('INFO')

    def get_Column_names(self):
        return self.column_names

    def parse(self, aeronet_file, skip_head=6, rows=7, padding=7):
        """
        Line:
        1: AERONET VERSION
        2: Place
        3: Version again, AOD Level
        5: Contact
        6: Notification
        7: Columns
        8:end - Data
        """
        self.reference = 550  # reference wavelength for turbidity calculation
        self.logger.info("Parsing %s" % aeronet_file)
        self.padding = padding
        frame = pd.read_csv(aeronet_file, header=skip_head, nrows=rows)
        self.column_names = np.concatenate((np.array(frame.columns), np.array([str(i) for i in range(padding)])))
        data = np.genfromtxt(aeronet_file, skip_header=7, delimiter=',', dtype=str)
        for idx, column in enumerate(self.column_names):
            self.aeronet_dict[column] = data[:, idx]
        return self.aeronet_dict

    def get_Timeline(self):
        time = self.aeronet_dict['Time(hh:mm:ss)']
        date = self.aeronet_dict['Date(dd-mm-yyyy)']
        concatenate_timeline = map(self.merge,zip(date, time))
        self.timeline = map(self.convert2datetime, concatenate_timeline)
        return self.timeline

    def show(self, key):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        self.get_Timeline()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.plot(self.timeline, self.aeronet_dict[key], '+')
        plt.gcf().autofmt_xdate()
        plt.show()

    def pretty_print(self, key=None):
        if key:
            print('%s' % self.aeronet_dict[key])
        else:
            for column in self.column_names[:-self.padding]:
                print("%s" % column)

    def get_turbidity(self):
        import matplotlib.pyplot as plt
        nm_in_strings = -2
        AOD_names = [aod for aod in self.aeronet_dict.keys() if aod.startswith('AOD')]
        lambdas = np.array([lam.split('_')[-1][:nm_in_strings] for lam in AOD_names])
        self.aeronet_dict['Turbidity'] = np.zeros(len(self.timeline))
        self.aeronet_dict['Turbidity_stderror'] = np.zeros(len(self.timeline))
        for idx, time in enumerate(self.timeline):
            aods_str = np.array([self.aeronet_dict[aod_name][idx] for aod_name in AOD_names])
            _idx = np.where(aods_str != '-999.000000')[0]
            wave_available = np.array(map(float, lambdas[_idx]))
            aods = np.array(map(float, aods_str[_idx]))
            sort_idx = wave_available.argsort()
            wave = wave_available[sort_idx][1:-3]  # magic numbers... to lazy right now to change this
            aods = aods[sort_idx][1:-3]  # magic numbers... to lazy right now to change this
            result = self._fit_aengstrom(aods, wave)
            wave_new = np.linspace(wave[0], wave[-1], 1000)
            fitted = self._aengstrom_formula(wave_new, result.params['alpha'].value, result.params['beta'].value)
            #plt = plot_turbidity(wave, aods, wave_new, fitted)
            self.aeronet_dict['Turbidity'][idx] = result.params['beta'].value
            self.aeronet_dict['Turbidity_stderror'][idx] = result.params['beta'].stderr
        self.logger.info("Available AODs %s " % sorted(wave_available))
        self.logger.info("ATTENTION badly hardcoded ============== Used AODs %s" % wave)


    def _aengstrom_formula(self, x, alpha, beta):
        return beta * (x / self.reference) ** (alpha)

    def _fit_aengstrom(self, Aods, lambdas):
        gmod = Model(self._aengstrom_formula, independent_vars=['x'], param_names=['alpha', 'beta'])
        gmod.set_param_hint('beta', value=0.1)
        gmod.set_param_hint('alpha', value=1.5)
        result = gmod.fit(Aods, x=lambdas)
        return result



if __name__ == "__main__":
    import argparse
    default_file = '/home/jana_jo/DLR/Codes/measurements/Aeronet/20161128_20161130_Munich_University/20161129_Munich_University.lev15'
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=default_file)
    args = parser.parse_args()
    Parser = AeronetParser()
    aeronaet = Parser.parse(args.file)
    time = Parser.get_Timeline()
    #Parser.show('Solar_Zenith_Angle(Degrees)')
    #Parser.pretty_print()
    Parser.get_turbidity()
    plt.plot(time, aeronaet['440-870_Angstrom_Exponent'], '+', label='440-870')
    #plt.errorbar(time, aeronaet['Turbidity'], yerr=aeronaet['Turbidity_stderror'], ecolor='g', fmt=None)
    plt.plot(time, aeronaet['380-500_Angstrom_Exponent'], '+', label='380-500')
    plt.plot(time, aeronaet['440-675_Angstrom_Exponent'], '+', label='440-675')
    plt.plot(time, aeronaet['500-870_Angstrom_Exponent'], '+', label='500-870')
    plt.legend()
    plt.show()
