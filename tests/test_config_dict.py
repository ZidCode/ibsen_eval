import numpy as np
from datetime import datetime


test_config = {'Processing': {'logging': False, 'logging_level': 'INFO'},
               'Data': {'dark':'../../measurements/Ostsee/T4/ST06/darkcurrent000.asc',
                        'utc_time': datetime(2016, 4, 14, 8, 47),
                        'gps_coords': np.array([ 53.9453236, 11.3829424,   0.]),
                        'target': '../../measurements/Ostsee/T4/ST06/target000.asc',
                        'reference': '../../measurements/Ostsee/T4/ST06/reference000.asc'}}

