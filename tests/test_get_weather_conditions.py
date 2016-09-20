import os
from datetime import datetime
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from evaluation.processing.get_weather_conditions import retrieve_weather_parameters, get_file_format, get_parameters


GPS_MUNICH = [48.08, 11.27]
UTC_TIME = datetime.strptime('2016-07-28 14:25:00', '%Y-%m-%d %H:%M:%S')


def test_file_format():
    download_file = get_file_format(GPS_MUNICH, UTC_TIME)
    working_dir = os.getcwd() + '/data/'
    assert download_file == working_dir + 'data_20160728_48.08,11.27.json'


def test_get_parameters():
    data = get_parameters(GPS_MUNICH, UTC_TIME)
    KEYS = np.array([u'current_observation', u'response', u'history'])
    assert_array_equal(KEYS, data.keys())


def test_weather_parameters():
    humidity_key = 'hum'
    pressure_key = 'pressurem'
    map_list = [humidity_key, pressure_key]
    map_dict = retrieve_weather_parameters(map_list, GPS_MUNICH, UTC_TIME)
    assert_almost_equal(map_dict[humidity_key], 0.56, decimal=1)
    assert map_dict[pressure_key] == 1017.0
    #Check downloading
    os.remove('data/data_20160728_48.08,11.27.json')
    humidity = retrieve_weather_parameters(map_list, GPS_MUNICH, UTC_TIME)
    assert_almost_equal(map_dict[humidity_key], 0.56, decimal=1)
