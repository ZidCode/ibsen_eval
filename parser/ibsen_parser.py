import numpy as np
import pandas as pd
import logging
import copy
from datetime import datetime
"""
FREEDOM VIS - Ibsen
    360 to 830 nm wavelength range
    Numerical aperture of 0.16
    Minimum resolution of 1.3 nm (FWHM)
    Footprint of 25 mm x 48 mm
ibsen_dict:
    {'num_of_meas': <int>,
    'data_mean': array([..]),
    'tdata': array([[..],..,[..]]) shape(30, 1024),
    'start_data_index': 18,
    'data_std': array([..]),
    'wave': array([..]),
    'IntTime': <float>,
    'data': array([[..],..,[..]]) shape(1024, 30)}
"""
flag_dict = {'[DataRaw]':False, '[DataCalibrated]':True}


def parse_ibsen_file(filename, maxrows=50):
    data_dict = dict()
    header = pd.read_csv(filename, nrows=maxrows, skip_blank_lines=False).values
    header = np.insert(header, 0, '[Measurement]')
    header_tmp = header.reshape(1, len(header))[0]
    # TODO: Add comment
    assert header.all() == header_tmp.all()
    data_dict['Type'] = header[np.where(np.array([str(s).find('Meas')
                               for s in header]) == 0)[0][0]].split()[-1]
    try:
        date =  header[np.where(np.array([str(s).find('Date') for s in header]) == 0)[0][0]].split()[-1]
        time =  header[np.where(np.array([str(s).find('UTCTime') for s in header]) == 0)[0][0]].split()[-1]
        data_dict['UTCTime'] = datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
    except IndexError:
        #logging.error("No [UTCTime] in measurments. Setting UTCTime to None")
        data_dict['UTCTime'] = None

    int_time = header[np.where(header == '[IntTime]')[0][0] + 1]
    data_dict['IntTime'] = np.array([float(inter) for inter in int_time.split()])
    data_dict['num_of_meas'] = len(data_dict['IntTime'])

    assert (data_dict['IntTime'][0] == data_dict['IntTime']).all(), 'Different Integrationtimes in file'
    data_dict['IntTime'] = data_dict['IntTime'][0]

    try:
        data_dict['start_data_index'] = np.where((header == '[DataRaw]') | (header == '[DataCalibrated]'))[0][0] + 1
        data_dict['darkcurrent_corrected'] = flag_dict[header[data_dict['start_data_index'] - 1]]
    except IndexError:
        logging.error('No [DataRaw] inside the header (TODO)')
    data = np.genfromtxt(filename, skip_header=data_dict['start_data_index'])
    data_dict['wave'] = data[:, 0]
    data_dict['data_mean'] = data[:, 1]
    data_dict['data_std'] = data[:, 2]
    data_dict['data_sample_std'] = np.std(data[:,3:], axis=1, ddof=1)
    data_dict['data'] = data[:, 3:]
    data_dict['tdata'] = np.transpose(data_dict['data'])
    data_dict['mean'] = get_mean_column(data_dict)
    return data_dict


def get_mean_column(ibsen_dict):
    # get mean columnwise tdata
    mean = np.mean(ibsen_dict['tdata'], axis=0)
    return mean

