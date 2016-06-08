import numpy as np
import pandas as pd
import logging
"""
FREEDOM VIS - Ibsen
    360 to 830 nm wavelength range
    Numerical aperture of 0.16
    Minimum resolution of 1.3 nm (FWHM)
    Footprint of 25 mm x 48 mm
"""


def parse_ibsen_file(filename, maxrows=50):
    data_dict = dict()
    header = pd.read_csv(filename, nrows=maxrows, skip_blank_lines=False).values
    header = np.insert(header, 0, '[Measurement]')
    header_tmp = header.reshape(1, len(header))[0]
    # TODO: Add comment
    assert header.all() == header_tmp.all()
    int_time = header[np.where(header == '[IntTime]')[0][0] + 1]
    data_dict['IntTime'] = np.array([float(inter) for inter in int_time.split()])
    data_dict['num_of_meas'] = len(data_dict['IntTime'])

    assert (data_dict['IntTime'][0] == data_dict['IntTime']).all(), 'Different Integrationtimes in file'
    data_dict['IntTime'] = data_dict['IntTime'][0]

    try:
        data_dict['start_data_index'] = np.where(header == '[DataRaw]')[0][0] + 1
    except IndexError:
        logging.error('No [DataRaw] inside the header (TODO)')
    data = np.genfromtxt(filename, skip_header=data_dict['start_data_index'])
    data_dict['wave'] = data[:, 0]
    data_dict['data_mean'] = data[:, 1]
    data_dict['data_std'] = data[:, 2]
    data_dict['data'] = data[:, 3:]
    data_dict['tdata'] = np.transpose(data_dict['data'])
    return data_dict


def subtract_dark_from_mean(tar, ref, dark):
    dark['mean'] = get_mean_column(dark)
    ref['mean'] = get_mean_column(ref)
    tar['mean'] = get_mean_column(tar)

    ref['mean'] -=  dark['mean']
    tar['mean'] -= dark['mean']

    ref['tdata'] -= dark['tdata']
    tar['tdata'] -= dark['tdata']

    assert dark['mean'].all() == np.mean(dark['data'], axis=1).all()


def get_mean_column(ibsen_dict):
    # get mean columnwise tdata
    mean = np.mean(ibsen_dict['tdata'], axis=0)
    return mean


def remove_outliers(data):
    #TODO
    pass
