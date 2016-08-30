import numpy as np
import copy
from numpy.testing import assert_equal, assert_array_equal
from tempfile import mkstemp
from evaluation.parser.ibsen_parser import parse_ibsen_file, subtract_dark_from_mean, get_mean_column


DEFAULT_KEYS = np.array(sorted(['num_of_meas', 'data_mean', 'tdata', 'start_data_index', 'data_std', 'wave', 'IntTime', 'data', 'Type', 'darkcurrent_corrected', 'data_sample_std']))


DEFAULT_MEAS = '[Measurement] \n\
Date    2016-05-25 \n\
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


def create_meas_file():
    file_ = mkstemp()

    filename = file_[1]
    with open(filename, 'w') as fp:
        fp.write(DEFAULT_MEAS)
    return filename


def test_parse_ibsen_file():
    filename = create_meas_file()
    ibsen_dict = parse_ibsen_file(filename)
    assert ibsen_dict['IntTime'] == 40, 'IntTime %s' % ibsen_dict['IntTime']
    assert ibsen_dict['Type'] == 'reference'
    assert_equal(np.array(sorted(ibsen_dict.keys())), DEFAULT_KEYS)


def test_substract_dark_from_mean():
    ref = create_meas_file()
    dark = create_meas_file()

    ref = parse_ibsen_file(ref)
    dark = parse_ibsen_file(dark)
    dark['Type'] = 'darkcurrent'

    ref_tmp = copy.deepcopy(ref)
    dark_tmp = copy.deepcopy(dark)
    ref_tmp['mean'] = get_mean_column(ref_tmp)
    dark_tmp['mean'] = get_mean_column(dark_tmp)
    assert ref['darkcurrent_corrected'] == False
    subtract_dark_from_mean(dark, ref)

    raw = ref_tmp['mean'] - dark_tmp['mean']
    assert ref['darkcurrent_corrected'] == True
    subtract_dark_from_mean(dark, ref)
    assert_array_equal(ref['mean'], ref_tmp['mean'] - dark_tmp['mean'])
    assert_array_equal(ref['mean'], np.zeros(2))
