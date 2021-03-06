import copy
import numpy as np
import pandas as pd
from BaseModels import BaseModelPython
from Model import SkyRadiance, LSkyRatio
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
    #setup['variables'] = ['alpha', 'beta', 'l_dsa', 'l_dsr', 'wv', 'H_oz']
    setup['variables'] = ['alpha', 'beta', 'l_dsa', 'l_dsr', 'g_dsr', 'g_dsa']
    setup['simulate']= [1.8, 0.06, 0.1, 0.17, 0.9,  0.8]
    setup['expected']= [1.8, 0.06, 0.1, 0.17]
    setup['guess'] = [1.5, 0.05, 0.1, 0.16]
    setup['bounds'] = [(-0.2, 5.), (0.0, 5), (0.01, 0.7), (0.01, 0.7)]  # config
    setup['model'] = LSkyRatio
    setup['noise'] = 0.0005  # 0.0005 for ratio 0.02 for L_sky
    setup['global'] = 'g_dsr'
    setup['local'] = 'g_dsa'
    setup['independent'] = {'x':np.linspace(350, 700, 1000), setup['global']:0, setup['local']:0}
    setup['dir'] = 'results/'
    return setup


def sensi_setup():
    sensi = dict()
    sensi['statistics'] = 20
    sensi['alpha'] = np.linspace(1.0, 2.5, 31)
    sensi['beta'] = np.linspace(0.04, 0.088, 31)
    sensi['l_dsr'] = np.linspace(0.15, 0.21, 51)
    sensi['l_dsa'] = np.linspace(0.01, 0.13, 31)
    sensi['g_dsa' ] = np.linspace(.5, 1., 31)
    sensi['g_dsr'] = np.linspace(0.5, 1.0, 31)
    sensi['H_oz'] = np.linspace(0.3, 0.51, 31)
    sensi['wv'] = np.linspace(0.1, 2.5, 31)
    return sensi


get_index = lambda x, l: np.where(x == np.array(l))[0][0]

def startsky2D(model_param, setup, sensi):
    r_setup = copy.copy(setup)

    model = BaseModelPython(model_param['zenith'], model_param['pressure'], model_param['ssa'])
    skyModel = setup['model'](model, model)
    kwargs = {key:value for key, value in zip(setup['variables'], setup['simulate'])}
    kwargs['x'] = setup['independent']['x']
    simulation = skyModel.func(**kwargs)
    import matplotlib.pyplot as plt
    plt.plot(kwargs['x'], simulation)
    plt.show()
    del model

    global_var = {'param': setup['global'], 'input': sensi[setup['global']], 'idx':get_index(setup['global'], setup['variables'])}
    local_var = {'param': setup['local'], 'input': sensi[setup['local']],'idx':get_index(setup['local'], setup['variables'])}

    for key in setup['independent'].keys():
        if key == 'x':
            # do nothing
            pass
        else:
            r_setup['variables'].remove(key)

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

        model = BaseModelPython(job_params['model_param']['zenith'], job_params['model_param']['pressure'], job_params['model_param']['ssa'])
        skyModel = job_params['setup']['model'](model, model)
        callable_ = skyModel.func

        r_setup = job_params['r_setup']
        local_var = job_params['local_var']
        fit_parameters = job_params['fit_params']
        global_var = job_params['global_var']
        setup = job_params['setup']
        input_value = job_params['global_var']['input']

        result = dict()
        frame = pd.DataFrame()

        for count in range(job_params['statistics']):
            noised_simulation = job_params['simulation'] + np.random.normal(0, setup['noise'], len(setup['independent']['x']))
            result['output'], result['success'] = _iterate(noised_simulation, r_setup['expected'], r_setup['guess'], r_setup['bounds'],
                                        r_setup['variables'], r_setup['independent'], callable_, job_params['global_var'], job_params['local_var'])
            for key, value in fit_parameters.items():
                frame.insert(0, value['columns'][count], result['output'][:, fit_parameters[key]['idx']])
            frame.insert(0, '%s_success' % count, result['success'])

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
        frame.to_csv('%stwo_variation_Deltaoutput_%s.txt' % (job_params['setup']['dir'], job_params['global_var']['file_index']), index=False)
        del model
        wq.task_done()


def _iterate(simulation, expected, guess, bounds, variables, independent, callable_, global_, local_):
    biased_parameters = np.zeros((len(local_['input']), len(expected)))
    success_ar = np.zeros(len(local_['input']))
    expected_dict = {key: value for key, value in zip(variables, expected)}
    gmod = Model(callable_, independent_vars=independent.keys(), param_names=variables, method='lbfgsb')
    for param, ini, limits in zip(variables, guess, bounds):
        gmod.set_param_hint(param, value=ini, min=limits[0], max=limits[1])

    independent[global_['param']] = global_['input']
    for i, input_ in enumerate(local_['input']):
        independent[local_['param']] = input_
        result = gmod.fit(simulation, verbose=False, **independent)
        print(result.fit_report())
        success_ar[i] = result.success
        biased_parameters[i] = np.array([result.params[key].value - expected_dict[key] for key in variables])
    return biased_parameters, success_ar


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
    model_param = get_model_param()
    setup = fit_setup()
    sensi = sensi_setup()
    startsky2D(model_param, setup, sensi)
