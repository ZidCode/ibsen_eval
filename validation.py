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
    Parser = AeronetParser()
    aeronaet = Parser.parse(config['source'])
    time = Parser.get_Timeline()
    Parser.get_turbidity()
    aeronaet['utc_times'] = time
    plot_aengstrom_parameters_aeronet(frame, aeronaet, title)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='Config')
    parser.add_argument('-r', '--results', help='Fitting results')
    parser.add_argument('-t', '--title', default='CHOOSE_TITLE_NAME', help='Set picture title')
    args = parser.parse_args()
    config = parse_ini_config(args.config)
    validate(config['Validation'], args.results, args.title)