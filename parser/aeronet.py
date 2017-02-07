#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from lmfit import Model
import ConfigParser
from utils.plotting import plot_turbidity



def parse_aeronet_config(ini_file):
    convert_to_array = lambda x, m : np.array([m(s) for s in x.split(',')])
    config = ConfigParser.ConfigParser()
    config.read(ini_file)
    config_dict = {s: dict(config.items(s)) for s in config.sections()}
    config_dict['Data']['aod_range'] = convert_to_array(config_dict['Data']['aod_range'], int)
    return config_dict['Data']


class AeronetParser:

    def __init__(self):
        self.reference = 550
        self.aeronet_dict = dict()
        self.convert2datetime = lambda d: datetime.strptime(d, '%d:%m:%Y %H:%M:%S')
        self.merge = lambda tup: tup[0] + ' ' + tup[1]
        self.padding = None
        self.timeline = None
        self.column_names = None
        self.exact_wavelength_columns = None
        self.logger = self._create_logger()

    def get_Column_names(self):
        return self.column_names

    def parse(self, aeronet_file, skip_head=6, rows=7, padding=7):
        self.logger.info("Parsing %s" % aeronet_file)
        self.padding = np.array([str(i) for i in range(padding)])
        frame = pd.read_csv(aeronet_file, header=skip_head, nrows=rows)
        columns = np.array(frame.columns)
        self.column_names = np.concatenate((columns, self.padding))
        self.exact_wavelength_columns = np.concatenate(([columns[-1]], self.padding))
        data = np.genfromtxt(aeronet_file, skip_header=7, delimiter=',', dtype=str)
        for idx, column in enumerate(self.column_names):
            self.aeronet_dict[column] = data[:, idx]
        return self.aeronet_dict

    def get_Timeline(self):
        time = self.aeronet_dict['Time(hh:mm:ss)']
        date = self.aeronet_dict['Date(dd-mm-yyyy)']
        concatenate_timeline = map(self.merge,zip(date, time))
        self.timeline = map(self.convert2datetime, concatenate_timeline)
        self.aeronet_dict['utc_times'] = self.timeline

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
            for column in self.column_names:
                print("%s" % column)

    def get_turbidity(self, aod_range):

        self.aeronet_dict['Turbidity'] = np.zeros(len(self.timeline))
        self.aeronet_dict['Turbidity_stderror'] = np.zeros(len(self.timeline))

        wave = self._get_exact_lambdas()
        AOD_names = self._delete_invalid_aods()
        assert len(AOD_names) == len(wave)
        self.logger.info("Lambdas %s \n AODs: %s" % (wave, AOD_names))
        wave, AOD_names = self._cut_range(aod_range, wave, AOD_names)
        self.logger.info("Used Lambdas %s \n \t \t Used AODs: %s" % (wave, AOD_names))
        angstrom_exponent_column = self._get_angstrom_columnname(aod_range)
        self.logger.info("Aengstrom Column %s" % angstrom_exponent_column)
        for idx, time in enumerate(self.timeline):
            aods = np.array([float(self.aeronet_dict[key][idx]) for key in AOD_names])
            self.alpha = float(self.aeronet_dict[angstrom_exponent_column][idx])
            result = self._fit_aengstrom(aods, wave)
            wave_new = np.linspace(wave[0], wave[-1], 1000)
            fitted = self._aengstrom_formula(wave_new, result.params['beta'].value)
            #plt = plot_turbidity(wave, aods, wave_new, fitted)
            self.aeronet_dict['Turbidity'][idx] = result.params['beta'].value
            self.aeronet_dict['Turbidity_stderror'][idx] = result.params['beta'].stderr
        #plt.show()

    def _get_angstrom_columnname(self, aod_range):
        return '%s-%s_Angstrom_Exponent' % (aod_range[0], aod_range[1])

    def _aengstrom_formula(self, x, beta):
        return beta * (x / self.reference) ** (-self.alpha)

    def _fit_aengstrom(self, Aods, lambdas):
        gmod = Model(self._aengstrom_formula, independent_vars=['x'], param_names=['beta'])
        gmod.set_param_hint('beta', value=0.1)
        result = gmod.fit(Aods, x=lambdas, verbose=False)
        return result

    def _get_exact_lambdas(self):
        return np.array(sorted([float(self.aeronet_dict[col][0]) * 1000 for col in self.exact_wavelength_columns]))

    def _delete_invalid_aods(self):
        nm_in_strings = -2
        AOD_names = np.array([aod for aod in self.aeronet_dict.keys() if aod.startswith('AOD')])
        lambdas = np.array([lam.split('_')[-1][:nm_in_strings] for lam in AOD_names])
        aods_str = np.array([self.aeronet_dict[aod_name][0] for aod_name in AOD_names])
        _idx = np.where(aods_str != '-999.000000')[0]
        lambdas = np.array(map(float, lambdas[_idx]))
        AOD_names = AOD_names[_idx]
        sort_idx = lambdas.argsort()
        return AOD_names[sort_idx]

    def _cut_range(self, aod_range, wave, AOD_names):
        start = 'AOD_%snm' % aod_range[0]
        end = 'AOD_%snm' % aod_range[1]
        idx_start = np.where(AOD_names == start)[0][0]
        idx_end = np.where(AOD_names == end)[0][0] + 1
        return wave[idx_start:idx_end], AOD_names[idx_start:idx_end]

    def _create_logger(self):
        logger = logging.getLogger('Aeronet')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s - %(funcName)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel('INFO')
        return logger


def main(config_file):
    config = parse_aeronet_config(config_file)

    Parser = AeronetParser()
    aeronaet = Parser.parse(config['source'])
    Parser.get_Timeline()
    #Parser.show('Solar_Zenith_Angle(Degrees)')
    #Parser.pretty_print()
    Parser.get_turbidity(config['aod_range'])
    #plt.plot(time, aeronaet['500-870_Angstrom_Exponent'], '+', label='440-870')
    plt.errorbar(aeronaet['utc_times'], aeronaet['Turbidity'], yerr=aeronaet['Turbidity_stderror'], ecolor='g', fmt=None)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='aeronet_config.ini')
    args = parser.parse_args()
    main(args.config)
