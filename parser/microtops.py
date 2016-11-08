import ConfigParser
import numpy as np
from datetime import datetime
from collections import OrderedDict

def med_time(utc_time, start, end):
    s = start.split(':')
    e = end.split(':')
    starttime = utc_time.replace(hour=int(e[0]), minute=int(e[1]))
    td = utc_time.replace(hour=int(s[0]), minute=int(s[1])) - starttime
    return starttime + td / 2

float_v = np.vectorize(float)
utc_calc = np.vectorize(med_time)

def extract_microtops_inifile(validation, utc_time):
    data = np.genfromtxt(validation['source'], skip_header=2, delimiter=',', dtype=str)

    micro_keys = ['utc_times', 'alpha', 'beta']
    micro_dict = {key: np.array([]) for key in micro_keys}

    micro_dict['alpha'] = float_v(data[:,2])
    micro_dict['alpha_stderr'] = float_v(data[:, 3])
    micro_dict['beta'] = float_v(data[:, 4])
    micro_dict['beta_stderr'] = float_v(data[:, 5])
    micro_dict['utc_times'] = utc_calc(utc_time, data[:,0], data[:,1])

    micro_dict['label'] = validation['label']
    return micro_dict


if __name__ == "__main__":
    validation = dict()
    validation['source'] = '/home/jana_jo/DLR/Codes/MicrotopsData/20160913_DLRRoof/aengstroem_results.txt'
    validation['label'] = 'microtops'
    UTCTime = datetime.strptime('2016-08-25 00:00:00', '%Y-%m-%d %H:%M:%S')
    print(extract_microtops_inifile(validation, UTCTime))