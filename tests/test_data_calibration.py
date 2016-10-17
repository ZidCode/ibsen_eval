import level0to1_processor as lp


#TODO
def test_calibrate_rawdata():
    nonlinearity_file = 'nonlinear.dat'
    response_file = 'ibsen_response.dat'
    raw_data = 'to_include.txt'
    calibrate_rawdata(raw_data, nonlinearity_file, response_file)
