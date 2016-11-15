import numpy as np
import copy
from datetime import datetime
from numpy.testing import assert_equal, assert_array_equal
from evaluation.parser.ibsen_parser import parse_ibsen_file, get_mean_column
from evaluation.utils.util import create_meas_file


DEFAULT_KEYS = np.array(sorted(['num_of_meas', 'data_mean', 'tdata', 'start_data_index', 'data_std', 'wave', 'IntTime', 'data', 'Type', 'darkcurrent_corrected', 'data_sample_std', 'UTCTime']))


DEFAULT_MEAS = '[Measurement] \n\
Date    2016-05-25 \n\
UTCTime 11:30:26 \n\
Project Radiometric Calibration \n\
Testsite    RASTA \n\
Station  \n\
MeasurementType reference \n\
NumSamples  30\n\
Comment Spectralon RASTA for calibration 40ms\n\
\n\
[SpectrometerHeader]\n\
Manufacturer    Ibsen\n\
Model   Freedom VIS FSV-305\n\
Detector    S10420-1006-01\n\
\n\
[IntTime]\n\
40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40  40 40  40  40  40  40  40  40  40  40  40  \n\
\n\
[DataRaw]\n\
313.22  1641.23 8.97    1636    1651    1635    1631    1635    1651    1647 \
1646    1649    1647    1632    1641    1644    1643    1650    1659    1644 \
1646    1626    1637    1637    1636    1649    1643    1641    1644    1629 \
1615    1653    1640   \n\
313.22  1641.23 8.97    1636    1651    1635    1631    1635    1651    1647 \
1646    1649    1647    1632    1641    1644    1643    1650    1659    1644 \
1646    1626    1637    1637    1636    1649    1643    1641    1644    1629 \
1615    1653    1640   \n'


def test_parse_ibsen_file():
    filename = create_meas_file(DEFAULT_MEAS)
    ibsen_dict = parse_ibsen_file(filename)
    UTCTime = datetime.strptime('2016-05-25 11:30:26', '%Y-%m-%d %H:%M:%S')
    assert ibsen_dict['IntTime'] == 40, 'IntTime %s' % ibsen_dict['IntTime']
    assert ibsen_dict['Type'] == 'reference'
    assert ibsen_dict['UTCTime'] == UTCTime
    assert_equal(np.array(sorted(ibsen_dict.keys())), DEFAULT_KEYS)
