import numpy as np
from datetime import datetime


test_config = {'Fitting': {'range_': np.array([0.35, 0.5]), 'alpha': 1.2, 'beta': 0.0},
               'Processing': {'logging': False, 'logging_level': 'INFO',
                              'gps_coords': np.array([ 53.9453236, 11.3829424,   0.]),
                              'utc_time': datetime(2016, 4, 14, 8, 47),
                              'params': ['hum', 'pressurem']},
               'Data': {'darkcurrent':'../../measurements/Ostsee/T4/ST06/darkcurrent000.asc',
                        'target': '../../measurements/Ostsee/T4/ST06/target000.asc',
                        'reference': '../../measurements/Ostsee/T4/ST06/reference000.asc'}}

