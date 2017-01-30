#!/usr/bin/env python
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


class AeronetParser:

    def __init__(self):
        self.aeronet_dict = dict()
        self.convert2datetime = lambda d: datetime.strptime(d, '%d:%m:%Y %H:%M:%S')
        self.merge = lambda tup: tup[0] + ' ' + tup[1]
        self.padding = None
        self.timeline = None
        self.column_names = None

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
        8:end Data
        """
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


if __name__ == "__main__":
    import argparse
    default_file = '/home/jana_jo/DLR/Codes/measurements/Aeronet/20161128_20161130_Munich_University/20161128_20161130_Munich_University.lev15'
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=default_file)
    args = parser.parse_args()
    Parser = AeronetParser()
    aeronaet = Parser.parse(args.file)
    time = Parser.get_Timeline()
    Parser.show('Exact_Wavelengths_of_AOD(um)')
    Parser.show('Solar_Zenith_Angle(Degrees)')
    Parser.pretty_print()
    plt.plot(time, aeronaet['440-870_Angstrom_Exponent'], '+', label='440-870')
    plt.plot(time, aeronaet['380-500_Angstrom_Exponent'], '+', label='380-500')
    plt.plot(time, aeronaet['440-675_Angstrom_Exponent'], '+', label='440-675')
    plt.plot(time, aeronaet['500-870_Angstrom_Exponent'], '+', label='500-870')
    # plt.plot(time, aeronaet['340-440_Angstrom_Exponent'])
    # plt.plot(time, aeronaet['440-675_Angstrom_Exponent[Polar]'])
    plt.legend()
    plt.show()
