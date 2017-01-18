#!/usr/bin/env python
import numpy as np
import pandas as pd
from datetime import datetime

date = lambda d: datetime.strptime(d, '%d:%m:%Y %H:%M:%S')

def parse_aeronet(aeronet_file):
    """
    Line:
    1: AERONET VERSION
    2: Place
    3: Version again, AOD Level
    4: bla bla
    5: Contact
    6: Notification
    7: Columns
    8:end Data
    """
    aeronet_dict = dict()
    frame = pd.read_csv(aeronet_file, header=6, nrows=7)
    column_names = np.concatenate((np.array(frame.columns), np.array([str(i) for i in range(7)])))
    data = np.genfromtxt(aeronet_file, skip_header=7, delimiter=',', dtype=str)
    #times =     time(strtim).replace(converteddatetime_newdate.year, newdate.month, newdate.day)
    for idx, column in enumerate(column_names):
        aeronet_dict[column] = data[:, idx]
    print(aeronet_dict.keys())
    print(aeronet_dict['380-500_Angstrom_Exponent'])
    print(aeronet_dict['500-870_Angstrom_Exponent'])


if __name__ == "__main__":
    import argparse
    default_file = '/home/jana_jo/DLR/Codes/measurements/Aeronet/test'
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=default_file)
    args = parser.parse_args()
    parse_aeronet(args.file)
