# ibsen_eval

evaluation.py :         evaluates reflectance deduced from target, reference and dark from the ibsen
TODO:files are still hardcoded. Should be passed by command line or stored in an ini-file for 
 with further configuration

parser/ibsen_parser.py:  parses ibsen files independent on ref/tar/dark (subtract_dark_from_mean and get mean included)
TODO: Instead of get_mean something like outlie detection method or sth. else should be implemented

parser/ibsen_calib....:  Calibrates ibsen data by means of RASTA 

python ibsen_calibration.py --help shows additional parameters
TODO: check_nonlinearity should be expanded by method correct_nonlinearity

processing/spectrum_analyser.py: Analyses spectrum (Reflectance)

processing/solar_zenith.py:      Extracts the solar zenith depending on GPS coordinates and utc_time

