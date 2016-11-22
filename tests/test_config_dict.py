import numpy as np
from datetime import datetime


test_config = {'Fitting': {'range_': np.array([350, 500]), 'params':['alpha', 'beta','g_dsa','g_dsr'],
                           'initial_values':np.array([1.2, 0.03, 0.128, 0.99]),
                           'limits':np.array([np.array([0.255,4]), np.array([0., 4.]), np.array([0.127, 0.129]), np.array([0.98, 1.])])},
               'Processing': {'logging': False, 'logging_level': 'INFO',
                              'gps_coords': np.array([ 53.9453236, 11.3829424,   0.]),
                              'utc_time': datetime(2016, 4, 14, 8, 47),
                              'params': ['hum', 'pressurem']},
               'Data': {'target': '../../measurements/Ostsee/T4/ST06/target000.asc',
                        'reference': '../../measurements/Ostsee/T4/ST06/reference000.asc'},
               'Validation': {'validate': True, 'source':'users/jana_jo/DLR/Codes/MicrotopsData/20160825_DLRRoof/results.ini',
                              'label': 'microtops'}}

