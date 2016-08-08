import os
from datetime import datetime
from numpy.testing import assert_almost_equal
from evaluation.processing.get_weather_conditions import retrieve_rel_humidity


def test_retrieve_rel_humidity():
    GPS_Munich = [48.08, 11.27]
    utc_time = datetime.strptime('2016-07-28 14:25:00', '%Y-%m-%d %H:%M:%S')
    humidity = retrieve_rel_humidity(GPS_Munich, utc_time)
    assert_almost_equal(humidity, 0.56, decimal=1)

    #Check downloading
    os.remove('data/data_20160728_48.08,11.27.json')
    humidity = retrieve_rel_humidity(GPS_Munich, utc_time)
    assert_almost_equal(humidity, 0.56, decimal=1)
