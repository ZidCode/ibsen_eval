import glob
import numpy as np
import pandas as pd
from evaluation import parse_ini_config, create_logger, evaluate_spectra


def retrieve(config, directory, output_file, logger):
    aided_params = dict()
    for key, aided_file in config['Aided']['params'].items():
        aided_params[key] = pd.read_csv(aided_file)

    file_prefixes = ['target', 'reference']
    files = [file_ for file_ in glob.iglob(directory + '%s*' % file_prefixes[0])]

    result_timeline = dict()
    for param in config['Fitting']['params']:
        result_timeline[param] = np.array([])
        result_timeline['%s_stderr' % param] = np.array([])
    result_timeline['sun_zenith'] = np.array([])
    result_timeline['utc_times'] = np.array([])

    for idx, file_ in enumerate(sorted(files)):
        for key in file_prefixes:
            config['Data'][key] = file_.replace(file_prefixes[0], key)
        try:
            logger.info("Evaluating file: %s \n" % file_)

            for key, frame in aided_params.items():
                config['Fitting']['independent'][key] = frame[key][idx]
                logger.info("========> Substitution %s with %s at time: %s" % (key, frame[key][idx], frame['utc_times'][idx]))

            params, result = evaluate_spectra(config, logger)
            result_timeline['utc_times'] = np.append(result_timeline['utc_times'], config['Processing']['utc_time']['tar'])
            assert set(aided_params.keys()).issubset(config['Fitting']['independent'].keys())

            while raw_input('Change initial (y or n)') == 'y':
                for idx, value in enumerate(config['Fitting']['initial_values']):
                    config['Fitting']['initial_values'][idx] = float(raw_input('Old: %s in %s ' % (value, idx)))
                params, result = evaluate_spectra(config, logger)

            result_timeline['sun_zenith'] = np.append(result_timeline['sun_zenith'], params['sun_zenith'])
            del params['sun_zenith']

            for key,item in params['variables'].items():
                result_timeline[key] = np.append(result_timeline[key], item['value'])
                stderr = '%s_stderr' % key
                result_timeline[stderr] = np.append(result_timeline[stderr], item['stderr'])

        except IOError:
            logger.error("%s have no corresponding reference" % file_)

    frame = pd.DataFrame(result_timeline)
    frame.to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='Pass ini-file for processing configurations')
    parser.add_argument('-m', '--measurement_directory', help='Define measurement directory to sweep through')
    parser.add_argument('-o', '--output_file', help='Write timeline results into output file')
    args = parser.parse_args()
    config = parse_ini_config(args.config)
    logger = create_logger(config['Processing'])
    retrieve(config, args.measurement_directory, args.output_file, logger)
