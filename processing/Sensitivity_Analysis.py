import copy
import numpy as np
import pandas as pd
from BaseModels import BaseModelPython
from Model import SkyRadianceSym, SkyRadiance
from get_ssa import get_ssa
from multiprocessing import Process, JoinableQueue
from multiprocessing import current_process
from atmospheric_mass import get_atmospheric_path_length
from lmfit import Model


def get_model_param():
    setup = dict()
    setup['zenith'] = 76.33134
    setup['AMass'] = get_atmospheric_path_length(setup['zenith'])
    setup['rel_h'] = 0.9
    setup['pressure'] = 950
    setup['AM'] = 5
    setup['ssa'] = get_ssa(setup['rel_h'], setup['AM'])
    return setup


def fit_setup():
    setup = dict()
    setup['variables'] = ['alpha', 'beta', 'l_dsr', 'l_dsa', 'H_oz', 'wv']
    setup['simulate']= [1.8, 0.06, 0.17, 0.1, 0.34, 1.2]
    setup['expected']= [1.8, 0.06, 0.1, 1.2]
    setup['guess'] = [1.5, 0.04, 0.09, 1.0]  # config
    setup['bounds'] = [(-0.2, 4), (0., 3),(-0.05, 0.3), (-0.05, 3.3)]  # config
    setup['model'] = SkyRadianceSym
    setup['global'] = 'H_oz'
    setup['local'] = 'l_dsr'
    setup['independent'] = {'x':np.linspace(650, 750, 1000), setup['global']:0, setup['local']:0}
    return setup


def sensi_setup():
    sensi = dict()
    sensi['statistics'] = 20
    sensi['alpha'] = np.linspace(1.0, 2.5, 51)
    sensi['beta'] = np.linspace(0.02, 0.13, 51)
    sensi['l_dsr'] = np.linspace(0.12, 0.2, 31)
    sensi['l_dsa'] = np.linspace(0.01, 0.1, 51)
    sensi['g_dsa' ] = np.linspace(.3, .8, 51)
    sensi['g_dsr'] = np.linspace(0.5, 1.0, 51)
    sensi['H_oz'] = np.linspace(0.3, 0.5, 31)
    sensi['wv'] = np.linspace(0.1, 2.4, 31)
    return sensi


get_index = lambda x, l: np.where(x == np.array(l))[0][0]

def startsky2D(logger):
    model_param = get_model_param()
    setup = fit_setup()
    sensi = sensi_setup()
    r_setup = copy.copy(fit_setup())

    model = BaseModelPython(model_param['zenith'], model_param['AMass'], model_param['pressure'], model_param['ssa'])
    skyModel = SkyRadiance(model)
    kwargs = {key:value for key, value in zip(setup['variables'], setup['simulate'])}
    kwargs['x'] = setup['independent']['x']
    simulation = skyModel.func(**kwargs)
    import matplotlib.pyplot as plt
    plt.plot(kwargs['x'], simulation)
    plt.show()
    del model

    global_var = {'param': setup['global'], 'input': sensi[setup['global']], 'idx':get_index(setup['global'], setup['variables'])}
    local_var = {'param': setup['local'], 'input': sensi[setup['local']],'idx':get_index(setup['local'], setup['variables'])}

    r_setup['variables'].remove(r_setup['global'])
    r_setup['variables'].remove(r_setup['local'])
    fit_parameters = dict()
    for idx, key in enumerate(r_setup['variables']):
        fit_parameters[key] = dict()
        fit_parameters[key]['idx'] = idx
        fit_parameters[key]['columns'] = ['%s_%s' %(key, i) for i in range(sensi['statistics'])]

    number_of_procs = 12
    work_q = JoinableQueue()
    process_list = [Process(target=parallel_sensi, args=(work_q,), name='sensi_proc_[%d]' % i) for i in range(number_of_procs)]
    for proc in process_list:
        proc.start()

    jobParam = dict()
    jobParam['cmd_type'] = 'EVAL'
    jobParam['model_param'] = model_param
    jobParam['fit_params'] = fit_parameters
    jobParam['local_var'] = local_var
    jobParam['setup'] = setup
    jobParam['global_var'] = global_var
    jobParam['r_setup'] = r_setup
    jobParam['simulation'] = simulation
    jobParam['statistics'] = sensi['statistics']

    for i, input_value in enumerate(global_var['input']):
        curJob = copy.deepcopy(jobParam)
        curJob['global_var']['input'] = input_value
        curJob['global_var']['file_index'] = i
        work_q.put(curJob)

    work_q.join()

    shutdownJob = dict()
    for i in range(number_of_procs):
        shutdownJob['cmd_type'] = 'SHUTDOWN'
        work_q.put(shutdownJob)

    work_q.join()

    print("Joining processes")
    for proc in process_list:
        proc.join()
    print("All done.")


def parallel_sensi(wq):
    while True:
        job_params = wq.get()
        if job_params['cmd_type'] == 'SHUTDOWN':
            print("%s is shutting down\n" % current_process().name)
            wq.task_done()
            break

        model = BaseModelPython(job_params['model_param']['zenith'], job_params['model_param']['AMass'], job_params['model_param']['pressure'], job_params['model_param']['ssa'])
        skyModel = SkyRadiance(model)
        callable = skyModel.func

        r_setup = job_params['r_setup']
        local_var = job_params['local_var']
        fit_parameters = job_params['fit_params']
        global_var = job_params['global_var']
        setup = job_params['setup']
        input_value = job_params['global_var']['input']

        result = dict()
        frame = pd.DataFrame()

        for count in range(job_params['statistics']):
            noised_simulation = job_params['simulation'] + np.random.normal(0, 0.04, len(setup['independent']['x']))
            result['output'] = _iterate(noised_simulation, r_setup['expected'], r_setup['guess'], r_setup['bounds'],
                                        r_setup['variables'], r_setup['independent'], callable, job_params['global_var'], job_params['local_var'])

            for key, value in fit_parameters.items():

                frame.insert(0, value['columns'][count], result['output'][:, fit_parameters[key]['idx']])
        for key, value in fit_parameters.items():
            mean = frame[value['columns']].iloc[:].mean(axis=1)
            stddev = frame[value['columns']].iloc[:].std(axis=1)
            frame.insert(0, '%s_std' % key, stddev)
            frame.insert(0, '%s_mean' % key, mean)

        frame.insert(0, local_var['param'], local_var['input'] - setup['simulate'][local_var['idx']])
        buf = np.zeros(len(local_var['input']))
        buf[0] = input_value - setup['simulate'][global_var['idx']]
        frame.insert(0, global_var['param'], buf)
        buf[0:len(setup['simulate'])] = setup['simulate']
        frame.insert(0, 'expected', buf)
        frame.to_csv('results/two_variation_Deltaoutput_%s.txt' % job_params['global_var']['file_index'], index=False)
        del model
        wq.task_done()


def _iterate(simulation, expected, guess, bounds, variables, independent, callable_, global_, local_):
    biased_parameters = np.zeros((len(local_['input']), len(expected)))
    expected_dict = {key: value for key, value in zip(variables, expected)}

    gmod = Model(callable_, independent_vars=independent.keys(), param_names=variables, method='lbfgsb')
    for param, ini, limits in zip(variables, guess, bounds):
        gmod.set_param_hint(param, value=ini, min=limits[0], max=limits[1])

    independent[global_['param']] = global_['input']
    for i, input_ in enumerate(local_['input']):
        independent[local_['param']] = input_
        result = gmod.fit(simulation, verbose=False, **independent)
        biased_parameters[i] = np.array([result.params[key].value - expected_dict[key] for key in variables])
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

