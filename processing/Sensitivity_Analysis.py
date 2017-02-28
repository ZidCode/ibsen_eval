import theano
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Model import IrradianceRatioSym, LSkyRatioSym, SkyRadianceSym
from FitModel import FitWrapper
from Residuum import Residuum
from get_ssa import get_ssa
from scipy.optimize import minimize, least_squares


def get_model_param():
    setup = dict()
    setup['zenith'] = 53.1836240528
    setup['AMass'] = 1.66450160404
    setup['rel_h'] = 0.665
    setup['pressure'] = 950
    setup['AM'] = 5
    setup['ssa'] = get_ssa(setup['rel_h'], setup['AM'])
    setup['x'] = np.linspace(350, 400, 1000)
    return setup


def fit_setup():
    setup = dict()
    setup['variables'] = ['alpha', 'beta', 'l_dsr', 'l_dsa', 'H_oz', 'wv']
    setup['simulate']= [1.8, 0.06, 0.07, 0.05, 0.34, 1.2]
    setup['expected']= [1.8, 0.06, 0.07, 0.05]
    setup['guess'] = [1.5, 0.04, 0.05, 0.02]  # config
    setup['bounds'] = [(-0.2, 4), (0., 3),(-0.05, 0.2), (-0.05, 0.1)]  # config
    setup['model'] = SkyRadianceSym
    setup['global'] = 'H_oz'
    setup['local'] = 'wv'
    return setup


def sensi_setup():
    sensi = dict()
    sensi['statistics'] = 20
    sensi['alpha'] = np.linspace(1.0, 2.5, 51)
    sensi['beta'] = np.linspace(0.02, 0.13, 51)
    sensi['l_dsr'] = np.linspace(0.01, 0.1, 51)
    sensi['l_dsa'] = np.linspace(0.01, 0.1, 51)
    sensi['g_dsa' ] = np.linspace(.3, .8, 51)
    sensi['g_dsr'] = np.linspace(0.5, 1.0, 51)
    sensi['H_oz'] = np.linspace(0.3, 0.5, 31)
    sensi['wv'] = np.linspace(0.1, 2.4, 31)
    return sensi


def start(logger):
    setup = fit_setup()
    model_param = get_model_param()
    sensi = sensi_setup()
    parameters = {value: idx for idx, value in enumerate(setup['variables'])}
    input_parameters = {'alpha': sensi['alpha'], 'beta': sensi['beta'],
                        'g_dsa': sensi['g_dsa'], 'g_dsr': sensi['g_dsr']}

    for rand in range(sensi['statistics']):
        for biased_parameter, biased_idx in parameters.items():
            result = dict()
            logger.debug("Simulation: %s" % setup['expected'])
            mu = setup['expected'][biased_idx]
            input_ = input_parameters[biased_parameter]

            irr_symbol = IrradianceRatioSym(model_param['zenith'], model_param['AMass'], model_param['pressure'], model_param['ssa'], model_param['x'])
            getIrrRatio = irr_symbol.get_compiled()
            simulation = getIrrRatio(*setup['expected'])  + np.random.normal(0, 0.001, len(model_param['x']))
            expected = copy.copy(setup['expected'])
            variables = copy.copy(setup['variables'])
            guess = copy.copy(setup['guess'])
            bounds = copy.copy(setup['bounds'])
            del variables[biased_idx]
            del expected[biased_idx]
            del guess[biased_idx]
            del bounds[biased_idx]

            result['output'] = _iterate(simulation, expected, guess, bounds, biased_parameter, irr_symbol, input_, logger)
            result['input'] = input_ - mu
            logger.info("======= Got %s" % result['output'])
            result['expected'] = np.zeros(len(input_))
            result['expected'][0:len(setup['expected'])] = setup['expected']
            frame = pd.DataFrame(result['output'], columns=variables)
            frame.insert(0, '%s' % biased_parameter, result['input'])
            frame.insert(0, 'expected', result['expected'])
            frame.to_csv('results/biased_%s_%s.txt' % (rand, biased_parameter), index=False)


def start2D(logger):
    model_param = get_model_param()
    setup = fit_setup()
    sensi = sensi_setup()
    r_setup = copy.copy(fit_setup())
    global_var = {'param': 'g_dsr', 'input': np.linspace(.752,.9, 51), 'idx':3}
    local_var = {'param': 'g_dsa', 'input': np.linspace(.4,.7, 51) ,'idx':2}
    for key in setup.keys():
        del r_setup[key][global_var['idx']]
        del r_setup[key][local_var['idx']]
    logger.info("here")

    irr_symbol = IrradianceRatioSym(model_param['zenith'], model_param['AMass'], model_param['pressure'], model_param['ssa'], model_param['x'])
    getIrrRatio = irr_symbol.get_compiled()
    simulation = getIrrRatio(*setup['expected'])

    fit_parameters = dict((key,0) for key in r_setup['variables'])
    map_parameters = {'alpha': 0, 'beta': 1}

    for key, value in fit_parameters.items():
        fit_parameters[key] = ['%s_%s' %(key, i) for i in range(sensi['statistics'])]
    for i, input_value in enumerate(global_var['input']):
        irr_symbol.setVariable(global_var['param'], input_value)
        logger.info(">>>> Global Input for %s: %s" %(global_var['param'], input_value))
        result = dict()
        frame = pd.DataFrame()
        for count in range(sensi['statistics']):

            noised_simulation = simulation + np.random.normal(0, 0.001, len(model_param['x']))
            logger.debug("Symbols %s" % irr_symbol.get_Symbols())
            logger.debug("Setup bounds:%s " % r_setup['bounds'])
            result['output'] = _iterate(noised_simulation, r_setup['expected'], r_setup['guess'], r_setup['bounds'], local_var['param'], irr_symbol, local_var['input'], logger,  2)
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
        frame.to_csv('new_results/two_variation_Deltaoutput_%s.txt' % i, index=False)
        logger.info(frame)
        logger.debug(r_setup)


get_index = lambda x, l: np.where(x == np.array(l))[0][0]

def startsky2D(logger):
    model_param = get_model_param()
    setup = fit_setup()
    sensi = sensi_setup()
    r_setup = copy.copy(fit_setup())
    print(model_param['zenith'])
    model = setup['model'](model_param['zenith'], model_param['AMass'], model_param['pressure'], model_param['ssa'], model_param['x'])
    callme = model.get_compiled()
    simulation = callme(*setup['simulate'])
    plt.plot(model_param['x'], simulation)
    plt.show()

    global_var = {'param': setup['global'], 'input': sensi[setup['global']], 'idx':get_index(setup['global'], setup['variables'])}
    local_var = {'param': setup['local'], 'input': sensi[setup['local']],'idx':get_index(setup['local'], setup['variables'])}

    r_setup['variables'].remove(r_setup['global'])
    r_setup['variables'].remove(r_setup['local'])
    fit_parameters = dict()
    for idx, key in enumerate(r_setup['variables']):
        fit_parameters[key] = dict()
        fit_parameters[key]['idx'] = idx
        fit_parameters[key]['columns'] = ['%s_%s' %(key, i) for i in range(sensi['statistics'])]

    for i, input_value in enumerate(global_var['input']):
        model.setVariable(global_var['param'], input_value)
        logger.info(">>>> Global Input for %s: %s" %(global_var['param'], input_value))
        result = dict()
        frame = pd.DataFrame()
        for count in range(sensi['statistics']):

            noised_simulation = simulation + np.random.normal(0, 0.001, len(model_param['x']))
            logger.debug("Setup bounds:%s " % r_setup['bounds'])
            result['output'] = _iterate(noised_simulation, r_setup['expected'], r_setup['guess'], r_setup['bounds'], local_var['param'], model, local_var['input'], logger)

            for key, value in fit_parameters.items():
                logger.debug("Inlude for parameter %s the column %s at index %s" % (key, value['columns'][count],  fit_parameters[key]['idx']))

                print(value['columns'][count])
                print(result['output'][:, fit_parameters[key]['idx']])
                frame.insert(0, value['columns'][count], result['output'][:, fit_parameters[key]['idx']])

        for key, value in fit_parameters.items():
            mean = frame[value['columns']].iloc[:].mean(axis=1)
            std = frame[value['columns']].iloc[:].std(axis=1)
            frame.insert(0, '%s_std' % key, std)
            frame.insert(0, '%s_mean' % key, mean)

        logger.debug("Local variable at place: %s" % local_var['idx'])
        logger.debug("Global variable at place: %s" % global_var['idx'])

        frame.insert(0, local_var['param'], local_var['input'] - setup['simulate'][local_var['idx']])

        buf = np.zeros(len(local_var['input']))
        buf[0] = input_value - setup['simulate'][global_var['idx']]
        frame.insert(0, global_var['param'], buf)
        print(len(local_var['input']))
        print(setup['expected'])
        buf[0:len(setup['simulate'])] = setup['simulate']
        frame.insert(0, 'expected', buf)
        frame.to_csv('results/two_variation_Deltaoutput_%s.txt' % i, index=False)
        logger.info(frame)
        logger.debug(r_setup)




def _iterate(simulation, expected, guess, bounds, biased_parameter, model, input_values, logger):
    biased_parameters = np.zeros((len(input_values), len(expected)))

    for i, input_ in enumerate(input_values):
        model.setVariable(biased_parameter, input_)
        logger.debug(">>>> Input for %s: %s" % (biased_parameter, input_))
        logger.debug("Fit variables %s: " % model.get_Symbols())
        res = Residuum(model)
        residuals = FitWrapper(res.getResiduum())
        resultls = minimize(residuals, guess, args=(simulation), jac=False, method='L-BFGS-B', bounds=bounds)
        logger.info("Fit : %s" % np.array(resultls.x))
        logger.info("Expected: %s" % expected)
        biased_parameters[i] = np.array(resultls.x) - np.array(expected)
        logger.info("Success: %s" %resultls.success)
    return biased_parameters


if __name__ == "__main__":
    import argparse
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='DEBUG')
    args = parser.parse_args()
    logger = logging.getLogger('sensivity_analysis')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(args.level)
    logger.info("Start")
    startsky2D(logger)
