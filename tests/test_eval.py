import os
import numpy as np
from datetime import datetime
import evaluation.evaluation as ie
from evaluation.parser.ibsen_parser import parse_ibsen_file


def test_evaluate():
    measurement = os.path.dirname(os.path.realpath(__file__)) + '/../../measurements/Ostsee/T2/ST06/'
    DEBUG=False
    gps_coords = [53.9453236, 11.3829424, 0]
    utc_time = datetime.strptime('2016-04-14 08:47:00', '%Y-%m-%d %H:%M:%S')
    files = ['reference000.asc', 'target000.asc','darkcurrent000.asc']
    file_set  = np.array([measurement + f for f in files])
    file_set = np.append(file_set, [gps_coords, utc_time, DEBUG])
    ie.evaluate(*file_set)

