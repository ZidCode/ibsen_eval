import ConfigParser
import numpy as np
from datetime import datetime
from collections import OrderedDict
import codecs


def med_time(utc_time, start, end):
    s = start.split(':')
    e = end.split(':')
    starttime = utc_time.replace(hour=int(e[0]), minute=int(e[1]))
    td = utc_time.replace(hour=int(s[0]), minute=int(s[1])) - starttime
    return starttime + td / 2

float_v = np.vectorize(float)
utc_calc = np.vectorize(med_time)


def parse_microtops_inifile(file_):
    micro_dict = dict()
    with codecs.open(file_, "r", encoding="utf-8-sig") as fp:
        UTC = fp.readline()
        UTCTime = datetime.strptime('%s 00:00:00' % UTC, '%d.%m.%Y %H:%M:%S')

    data = np.genfromtxt(file_, skip_header=1, delimiter=',', dtype=str)
    micro_dict['utc_times'] = utc_calc(UTCTime, data[1:,0], data[1:,1])

    for idx, key in enumerate(data[0, 2:]):
        micro_dict[key] = float_v(data[1:, idx + 2])
    micro_dict['label'] = 'microtops'
    return micro_dict


if __name__ == "__main__":
    filename =  '/home/jana_jo/DLR/Codes/MicrotopsData/20160913_DLRRoof/aengstroem_results.txt'
    UTCTime = datetime.strptime('2016-08-25 00:00:00', '%Y-%m-%d %H:%M:%S')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='DEBUG')
    args = parser.parse_args()
    m = parse_microtops_inifile(args.file)
    print(m['beta'])
# /home/jana_jo/DLR/Codes/measurements/MicrotopsData/29_11_2016/380_870nm/aengstrom_wv_ozone_results_380_870.txt