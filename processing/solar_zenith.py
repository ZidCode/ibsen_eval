import ephem
import numpy as np


def get_sun_zenith(utc_time, lat, lon, el=0.0):
    detect = ephem.Observer()
    detect.lat = str(lat)
    detect.lon = str(lon)
    detect.elevation = el
    detect.date = utc_time
    sun = ephem.Sun(detect)
    sun.compute(detect)
    return 90 - np.degrees(float(sun.alt ))
