from datetime import datetime
from evaluation.parser.microtops import extract_microtops_inifile


def test_extract_microtops_inifile():
    validation = dict()
    validation['label'] = 'test'
    validation['source']='/home/jana_jo/DLR/Codes/MicrotopsData/20160825_DLRRoof/aengstroem_results.txt'
    utc_date = datetime.strptime('2016-08-25 10:00:30', '%Y-%m-%d %H:%M:%S')
    result = extract_microtops_inifile(validation, utc_date)
    UTC = datetime.strptime('2016-08-25 10:41:00', '%Y-%m-%d %H:%M:%S')
    print(result)
    assert result['utc_times'][1] == UTC
    assert result['alpha'][0] == 1.58
