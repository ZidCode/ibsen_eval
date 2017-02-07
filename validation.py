import pandas as pd
from datetime import datetime
from evaluation import parse_ini_config
from parser.microtops import parse_microtops_inifile
from utils.plotting import plot_aengstrom_parameters, plot_aengstrom_parameters_aeronet
from parser.aeronet import AeronetParser

convert2datetime = lambda d: datetime.strptime(d, '%Y-%m-%d %H:%M:%S')



def validate(config, results, title):
    frame = pd.read_csv(results)
    frame['utc_times'] = [convert2datetime(utc) for utc in frame['utc_times']]
    validation = dict()
    # Aeronet
    Parser = AeronetParser()
    aeronaet = Parser.parse(config['aeronet'])
    UTCTime = datetime.strptime('2016-11-29 00:00:00', '%Y-%m-%d %H:%M:%S')
    #Microtops
    micro_dict = parse_microtops_inifile(config['micro'], UTCTime)
    time = Parser.get_Timeline()
    Parser.get_turbidity()
    aeronaet['utc_times'] = time
    plot_aengstrom_parameters_aeronet(frame, aeronaet, micro_dict, title)


def validate(config):
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='Config')
    args = parser.parse_args()
    config = parse_ini_config(args.config)
    validate(config['Validation'])
