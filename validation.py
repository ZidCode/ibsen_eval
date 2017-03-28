from evaluation import parse_ini_config
from utils.plotting import plot_aengstrom_parameters_aeronet, micro_plot, ibsen_plot, aeronet_plot, plot_wv_ozone, ibsen_wv_plot, aeronet_wv_plot, micro_wv_plot
from parser.aeronet import AeronetParser
from parser.microtops import parse_microtops_inifile
import pandas as pd
from utils.util import convert2datetime


def validate(config):

    PlotWrapper = ValidationFactory(config)
    obj_list = PlotWrapper()
    plot_aengstrom_parameters_aeronet(obj_list, config['title'])


def validate_wv_ozone(config):
    PlotWrapper = ValidationFactory(config)
    obj_list = PlotWrapper()
    plot_wv_ozone(obj_list, config['title'])


class ValidationFactory:

    def __init__(self, config):
        self.config = config

    def __call__(self):
        map_validation = {'results': IbsenPlot, 'micro': MicroPlot, 'aeronet': AeronetPlot}
        obj_list = [value(self.config[key], self.config['aod_range']) for key, value in map_validation.items() if key in self.config['validate']]
        return obj_list


class IbsenPlot:

    def __init__(self, source, _):
        print("IbsenPlot Constructor")
        self.misc_frame = pd.DataFrame()

        for key, source_file in source.items():
            frame =  pd.read_csv(source_file)
            print("%s key % source" %(key, source_file))
            self.misc_frame[key] = frame[key]
            self.misc_frame['%s_stderr' % key] = frame['%s_stderr' % key]
            self.misc_frame['utc_times'] = [convert2datetime(utc) for utc in frame['utc_times']]

    def get_wv_plot(self, ax):
        return ibsen_wv_plot(self.misc_frame, *ax)

    def get_plot(self, ax):
        return ibsen_plot(self.misc_frame, *ax)


class AeronetPlot:

    def __init__(self, source, aod_range):
        print("AeronetPlot Constructor")
        Parser = AeronetParser()
        self.aeronet = Parser.parse(source)
        Parser.get_Timeline()
        Parser.get_turbidity(aod_range)

    def get_wv_plot(self, ax):
        return aeronet_wv_plot(self.aeronet, *ax)

    def get_plot(self, ax):
        return aeronet_plot(self.aeronet, *ax)


class MicroPlot:

    def __init__(self, source, _):
        print("MicroPlot Constructor")
        self.micro_dict = parse_microtops_inifile(source)

    def get_wv_plot(self, ax):
        return micro_wv_plot(self.micro_dict, *ax)

    def get_plot(self, ax):
        return micro_plot(self.micro_dict, *ax)


class PlotWrapper:
    def __init__(self, object_list):
        self.objs = object_list

    def get_plot(self):
        return [obj.get_plot() for obj in self.objs]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='Config')
    parser.add_argument('-wv', '--wv_ozone', default=False, action='store_true')
    args = parser.parse_args()
    config = parse_ini_config(args.config)
    if args.wv_ozone:
        validate_wv_ozone(config['Validation'])
    else:
        validate(config['Validation'])
