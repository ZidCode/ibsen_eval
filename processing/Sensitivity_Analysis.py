import theano
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Model import IrradianceModel, TmpModel
from FitModel import FitWrapper, FitModel
from irradiance_models import irradiance_models
from Residuum import Residuum
from get_ssa import get_ssa

""" Spaghetti time"""


def get_model_param():
    setup = dict()
    setup['zenith'] = 53.18
    setup['AMass'] = 1.664
    setup['rel_h'] = 0.665
    setup['pressure'] = 950
    setup['AM'] = 5
    setup['ssa'] = get_ssa(setup['rel_h'], setup['AM'])
    setup['x'] = np.linspace(350, 700, 100)
    return setup


def set_up():
    setup = dict()
    variables = ['alpha', 'beta', 'g_dsa', 'g_dsr']
    setup['variables'] = variables
    setup['expected']= [1.8, 0.06, 0.32, 0.32]
    setup['guess'] = [1.0, 0.01, 0.2, 0.2]  # config
    setup['bounds'] = [(-0.2, 4), (0., 3), (0., 1.), (0., 1.)]  # config
    return setup


def start(logger):
    setup = set_up()
    model_param = get_model_param()
    parameters = {'alpha': 0, 'beta': 1, 'g_dsa': 2, 'g_dsr': 3}
    input_parameters = {'alpha': np.linspace(1.0, 2.5, 101), 'beta': np.linspace(0.02, 0.1, 101),
                        'g_dsa': np.linspace(.1, .5, 101), 'g_dsr': np.linspace(0.1, .5, 101)}

    for rand in range(30):
        for biased_parameter, biased_idx in parameters.items():
            result = dict()
            logger.debug("Simulation: %s" % setup['expected'])
            mu = setup['expected'][biased_idx]
            input_ = input_parameters[biased_parameter]

            irr_symbol = IrradianceModel(model_param['x'], model_param['zenith'], model_param['AMass'], model_param['pressure'], model_param['ssa'])
            getIrrRatio = irr_symbol.getcompiledModel('ratio')
            simulation = getIrrRatio(*setup['expected'])  + np.random.normal(0, 0.001, len(model_param['x']))
            expected = copy.copy(setup['expected'])
            variables = copy.copy(setup['variables'])
            guess = copy.copy(setup['guess'])
            bounds = copy.copy(setup['bounds'])
            del variables[biased_idx]
            del expected[biased_idx]
            del guess[biased_idx]
            del bounds[biased_idx]

            result['output'] = _iterate(simulation, variables, expected, guess, bounds, biased_parameter, irr_symbol, input_, logger)
            result['input'] = input_ - mu
            logger.info("======= Got %s" % result['output'])
            result['expected'] = np.zeros(len(input_))
            result['expected'][0:len(setup['expected'])] = setup['expected']
            frame = pd.DataFrame(result['output'], columns=variables)
            frame.insert(0, '%s' % biased_parameter, result['input'])
            frame.insert(0, 'expected', result['expected'])
            frame.to_csv('results/single_analyse_kak2D/biased_%s_%s.txt' % (rand, biased_parameter), index=False)


def start2D(logger):
    model_param = get_model_param()
    setup = set_up()
    r_setup = copy.copy(set_up())
    noise_count = 30
    global_var = {'param': 'g_dsr', 'input': np.linspace(.16,.5,41), 'idx':3}
    local_var = {'param': 'g_dsa', 'input': np.linspace(.1,.5,41) ,'idx':2}
    for key in setup.keys():
        del r_setup[key][global_var['idx']]
        del r_setup[key][local_var['idx']]

    irr_symbol = IrradianceModel(model_param['x'], model_param['zenith'], model_param['AMass'], model_param['pressure'], model_param['ssa'])
    getIrrRatio = irr_symbol.getcompiledModel('ratio')
    simulation = getIrrRatio(*setup['expected'])

    fit_parameters = dict((key,0) for key in r_setup['variables'])
    map_parameters = {'alpha': 0, 'beta': 1}
    for key, value in fit_parameters.items():
        fit_parameters[key] = ['%s_%s' %(key, i) for i in range(noise_count)]

    for i, input_value in enumerate(global_var['input']):
        irr_symbol.setVariable(global_var['param'], input_value)
        logger.info(">>>> Global Input for %s: %s" %(global_var['param'], input_value))
        result = dict()
        frame = pd.DataFrame()
        for count in range(noise_count):

            noised_simulation = simulation + np.random.normal(0, 0.001, len(model_param['x']))
            logger.debug("Symbols %s" % irr_symbol.get_Symbols())
            result['output'] = _iterate(noised_simulation, [], r_setup['expected'], r_setup['guess'], r_setup['bounds'], local_var['param'], irr_symbol, local_var['input'], logger,  2)
            for key, value in fit_parameters.items():
                logger.debug("Inlude for parameter %s the column %s at index %s" % (key, value[count], map_parameters[key]))
                frame.insert(0, value[count], result['output'][:, map_parameters[key]])

        for key, value in fit_parameters.items():
            mean = frame[value].iloc[:].mean(axis=1)
            std = frame[value].iloc[:].std(axis=1)
            frame.insert(0, '%s_std' % key, std)
            frame.insert(0, '%s_mean' % key, mean)

        frame.insert(0, local_var['param'], local_var['input'] - setup['expected'][local_var['idx']])
        buf = np.zeros(len(local_var['input']))
        buf[0] = input_value - setup['expected'][global_var['idx']]
        frame.insert(0, global_var['param'], buf)
        buf[0:len(setup['expected'])] = setup['expected']
        frame.insert(0, 'expected', buf)
        frame.to_csv('results/two_variation_Deltaoutput_%s.txt' % i, index=False)
        logger.info(frame)
        logger.debug(r_setup)


def _iterate(simulation, variables, expected, guess, bounds, biased_parameter, model, input_values, logger, len_parameter=3):
    biased_parameters = np.zeros((len(input_values), len_parameter))
    for i, input_ in enumerate(input_values):
        model.setVariable(biased_parameter, input_)
        logger.debug(">>>> Input for %s: %s" % (biased_parameter, input_))
        logger.debug("Fit variables %s: " % model.get_Symbols())
        res = Residuum(model, 'ratio')
        residuals = FitWrapper(res.getResiduals())
        Fit = FitModel()
        resultls = Fit._least_squares(residuals, guess, simulation, bounds)
        biased_parameters[i] = np.array(resultls.x) - np.array(expected)
    return biased_parameters


if __name__ == "__main__":
    import argparse
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='INFO')
    args = parser.parse_args()
    logger = logging.getLogger('sensivity_analysis')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(args.level)
    #start(logger)
    start2D(logger)
