import ephem
from datetime import datetime
from numpy.testing import assert_approx_equal
from evaluation.processing.solar_zenith import get_sun_zenith


def test_get_sun_zenith():
    # Reference http://www.esrl.noaa.gov/gmd/grad/solcalc/azel.html
    # http://www.esrl.noaa.gov/gmd/grad/solcalc/
    # Solar elevation 57.76
    obs_lon = 11.2738613
    obs_lat = 48.085148
    utc_time = datetime.strptime('2016-07-11 12:55:25', '%Y-%m-%d %H:%M:%S')
    angle = get_sun_zenith(utc_time, obs_lat, obs_lon)
    assert_approx_equal(angle, 90 - 57.76, significant=2)
