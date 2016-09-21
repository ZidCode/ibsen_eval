import ConfigParser
import numpy as np
from collections import OrderedDict


def extract_microtops_inifile(source, utc_time):
    config = ConfigParser.ConfigParser()
    config.read(source)
    config_dict = {s: dict(config.items(s)) for s in config.sections()}
    micro_keys = ['utc_times', 'alpha', 'beta']
    micro_dict = {key: np.array([]) for key in micro_keys}
    tmp_dict = dict()
    for key, item in sorted(config_dict.items()):
        e = item['starttime'].split(':')
        s = item['endtime'].split(':')
        starttime = utc_time.replace(hour=int(e[0]), minute=int(e[1]), second=int(e[2]))
        td = utc_time.replace(hour=int(s[0]), minute=int(s[1]), second=int(s[2])) - starttime
        utc_time =  starttime + td / 2
        tmp_dict[utc_time] = dict()
        tmp_dict[utc_time]['alpha'] = item['alpha']
        tmp_dict[utc_time]['beta'] = item['beta']

    tmp_dict = OrderedDict(sorted(tmp_dict.items(), key=lambda t: t[0]))
    for key, value in tmp_dict.items():
        micro_dict['utc_times'] = np.append(micro_dict['utc_times'], key)
        micro_dict['alpha'] = np.append(micro_dict['alpha'], float(value['alpha']))
        micro_dict['beta'] = np.append(micro_dict['beta'], float(value['beta']))
    return micro_dict
