from datetime import datetime
from evaluation.parser.microtops import extract_microtops_inifile


def test_extract_microtops_inifile():
    source='/users/jana_jo/DLR/Codes/MicrotopsData/20160825_DLRRoof/results.ini'
    utc_date = datetime.strptime('2016-08-25 10:00:30', '%Y-%m-%d %H:%M:%S')
    result = extract_microtops_inifile(source, utc_date)
    UTC = datetime.strptime('2016-08-25 10:40:30', '%Y-%m-%d %H:%M:%S')
    assert result['utc_times'][1] == UTC
    assert result['alpha'][0] == 1.78
