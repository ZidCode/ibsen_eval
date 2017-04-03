import os
import re
import urllib2
import json
import copy
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import interpolate
from api_key import api_key
from matplotlib.font_manager import FontProperties


FONTSTYLE = 'serif'
FONTSIZE = 12
hfont = {'family':FONTSTYLE, 'fontsize': FONTSIZE}
fontP = FontProperties()
fontP.set_family(FONTSTYLE)
fontP.set_size('small')


def download_weather_data(destiny):
    """
    Destiny  data_%date_%gps.json
    http://api.wunderground.com/api/%API_KEY/history_20160728/conditions/q/49.46,11.18.json
    """
    utc_gps = re.findall(r"\d+[\.\d+]*", destiny.split('/')[-1])
    url = 'http://api.wunderground.com/api/%s/history_%s/conditions/q/%s,%sjson' %(api_key, utc_gps[0], utc_gps[1], utc_gps[2])

    f = urllib2.urlopen(url)
    json_string = f.read()
    parsed_json = json.loads(json_string)
    with open(destiny, 'w') as fp:
        json.dump(parsed_json, fp)


def get_file_format(gps, utc_time):
    dir_ = os.environ['PYTHONPATH'].split(':')[-1]
    data_folder = dir_ + '/tests/data/'
    gps = '%s,%s' %(gps[0], gps[1])
    utc = utc_time.strftime("%Y%m%d")
    file_format = data_folder + 'data_' + utc + '_' + gps + '.json'
    return file_format


def get_parameters(gps, utc_time):
    json_file = get_file_format(gps, utc_time)
    try:
        with open(json_file, 'r') as fp:
            data = json.load(fp)
    except IOError:
        print("Info: File not present. Start downloading")
        download_weather_data(json_file)
        # Duplicated code (alternative)
        with open(json_file, 'r') as fp:
            data = json.load(fp)

    return data


def extract_ts(params, data, utc_time):
    """
    Args:
        params: list of parameters to extract from weather conditions
        data:   parsed dictionary from json get_parameter
        utc_time: demanded utc_time
    Return:
        param_ts: Dictionary with time_series and corresponding parameter series
    """
    params_ts = dict()
    fl_vector = np.vectorize(float)
    utc_tmp = copy.copy(utc_time)
    param_ts = {key: np.array([]) for key in params}
    param_ts['time'] = np.array([])

    for his in data['history']['observations']:
        tmp = copy.copy(utc_time)
        tmp = tmp.replace(hour=int(his['utcdate']['hour']), minute=int(his['utcdate']['min']))
        param_ts['time'] = np.append(param_ts['time'], tmp)
        for param in params:
            param_ts[param] = np.append(param_ts[param], his[param])

    return param_ts


# TODO: Use decorators..
def retrieve_weather_parameters(params, gps, utc_time, debug=False):
    scaling_properties = {'hum': lambda x: x / 100, 'pressurem': lambda x: x}
    utc_param_values = dict()
    inter = dict()
    #Retrieving weather condition parameters
    data = get_parameters(gps, utc_time)

    #Retrieven series
    param_ts = extract_ts(params, data, utc_time)

    history_stamps = mdates.date2num(param_ts['time'])
    utc_stamp = mdates.date2num(utc_time)
    for param in params:
        inter[param] = interpolate.interp1d(history_stamps, param_ts[param])
        utc_param_values[param] = scaling_properties[param](inter[param](utc_stamp))


    if debug:
        t_new = np.linspace(min(history_stamps), max(history_stamps), 1000)
        fix, ax1 = plt.subplots()

        ax1.plot(param_ts['time'], param_ts['hum'], '+')
        ax1.plot(utc_stamp, utc_param_values['hum'] * 100, 'o', label=r'rel. hum %.1f' % utc_param_values['hum'] )
        ax1.plot(t_new, inter['hum'](t_new), 'b')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('rel. humidity %')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        ax2 = ax1.twinx()
        ax2.plot(param_ts['time'], param_ts['pressurem'], '+')
        ax2.plot(utc_stamp, utc_param_values['pressurem'], 'o', label=r'rel. press %.1f' % utc_param_values['pressurem'])
        ax2.plot(t_new, inter['pressurem'](t_new))

        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        ax2.set_ylabel('Pressure')
        plt.title(utc_time.strftime("Day %d.%m.%Y") )
        plt.legend()
        plt.show()

    return utc_param_values


def show_more_days():
    params = ['hum', 'pressurem']
    gps = [48.14800, 11.57300]
    utc_time1 = datetime.strptime('2017-02-16 10:43:20', '%Y-%m-%d %H:%M:%S')
    utc_time2 = datetime.strptime('2016-11-29 10:43:20', '%Y-%m-%d %H:%M:%S')
    utc_time3 = datetime.strptime('2016-12-06 10:43:20', '%Y-%m-%d %H:%M:%S')
    scaling_properties = {'hum': lambda x: x / 100, 'pressurem': lambda x: x}

    t_new = []
    fix, ax1 = plt.subplots()
    utc_times = [utc_time1, utc_time2, utc_time3]
    markers = ['v', 'o', 'p']
    for mark, utc_time in zip(markers, utc_times):
        utc_param_values = dict()
        inter = dict()
        data = get_parameters(gps, utc_time)
        param_ts = extract_ts(params, data, utc_time)
        lable = '%s.%s.%s' % ('{:02d}'.format(param_ts['time'][0].day), '{:02d}'.format(param_ts['time'][0].month), param_ts['time'][0].year)
        for idx, time in enumerate(param_ts['time']):
            param_ts['time'][idx] = time.replace(year=2016, month=11, day=29)
        ax1.plot(param_ts['time'], param_ts['hum'], label=lable, marker=mark,  linestyle = 'None')

    ax1.set_xlabel('UTC Timestamp', **hfont)
    ax1.set_ylabel(r'rel. humidity $\left[ \% \right]$', **hfont)
    plt.legend(loc='best', prop=fontP, title='Date')
    plt.show()


if __name__ == '__main__':
    show_more_days()
    #GPS_Munich = [48.14800, 11.57300]
    #utc_time = datetime.strptime('2017-02-16 10:43:20', '%Y-%m-%d %H:%M:%S')
    #params = ['hum', 'pressurem']
    #vals = retrieve_weather_parameters(params, GPS_Munich, utc_time, True)
    #print("Rel. Hum: %s and Pressure %s" % (vals['hum'], vals['pressurem']))
