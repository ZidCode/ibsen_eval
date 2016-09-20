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


def extract_humidity(data, utc_time):
    fl_vector = np.vectorize(float)
    utc_tmp = copy.copy(utc_time)
    time_series = np.array([])
    humidity = np.array([])
    pressure = np.array([])

    for his in data['history']['observations']:
        tmp = copy.copy(utc_time)
        tmp = tmp.replace(hour=int(his['utcdate']['hour']), minute=int(his['utcdate']['min']))
        time_series = np.append(time_series, tmp)
        humidity = np.append(humidity, his['hum'])
        pressure = np.append(pressure, his['pressurem'])

    return fl_vector(humidity), fl_vector(pressure), time_series


# TODO: Use decorators..
def retrieve_rel_humidity(gps, utc_time, debug=False):
    data = get_parameters(gps, utc_time)
    humidity, pressure, time_series = extract_humidity(data, utc_time)
    history_stamps = mdates.date2num(time_series)
    utc_stamp = mdates.date2num(utc_time)
    inter = interpolate.interp1d(history_stamps, humidity)
    inter_press = interpolate.interp1d(history_stamps, pressure)
    utc_humidity_value = inter(utc_stamp)
    utc_pressure_value = inter_press(utc_stamp)
    if debug:
        t_new = np.linspace(min(history_stamps), max(history_stamps), 1000)
        fix, ax1 = plt.subplots()

        ax1.plot(time_series, humidity, '+')
        ax1.plot(utc_stamp, utc_humidity_value, 'o', label=r'rel. hum %.1f' % utc_humidity_value)
        ax1.plot(t_new, inter(t_new), 'b')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('rel. humidity %')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        ax2 = ax1.twinx()
        ax2.plot(time_series, pressure, '+')
        ax2.plot(utc_stamp, utc_pressure_value, 'o', label=r'rel. press %.1f' % utc_pressure_value)
        ax2.plot(t_new, inter_press(t_new))

        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        ax2.set_ylabel('Pressure')
        plt.title(utc_time.strftime("Day %d.%m.%Y") )
        plt.legend()
        plt.show()
    return utc_humidity_value / 100, utc_pressure_value


if __name__ == '__main__':
    GPS_Munich = [48.08, 11.27]
    utc_time = datetime.strptime('2016-07-28 14:25:00', '%Y-%m-%d %H:%M:%S')
    humidity = retrieve_rel_humidity(GPS_Munich, utc_time, True)

