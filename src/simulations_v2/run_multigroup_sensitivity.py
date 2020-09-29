import argparse
import multiprocessing
import os
import pdb
import socket
import time
import uuid
import yaml

import dill
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
import sqlalchemy

from analysis_helpers import run_multiple_trajectories
import db_config
from multi_group_simulation import MultiGroupSimulation
from load_multigroup_params import load_params
from run_sensitivity import create_scenario_dict

BASE_DIRECTORY = os.path.abspath(os.path.join('')) + "/sim_output/"

VALID_GLOBAL_PARAMS_TO_VARY = [
    'contact_tracing_isolations',
    'expected_contacts_per_day',
    'symptomatic_daily_self_report_p',
    'asymptomatic_daily_self_report_p',
    'initial_ID_prevalence',
    'exposed_infection_p',
    'asymptomatic_p',
    'contact_tracing_delay',
    'test_protocol_QFNR',
    'test_population_fraction',
    'daily_outside_infection_p',
    'initial_E_count',
    'initial_pre_ID_count',
    'initial_ID_count',
    'initial_SyID_mild_count',
    'initial_SyID_severe_count',
    'population_size'
    ]


def simulate(args):
    """
    Main function that initializes the simulations, executes them, and then
    manages the results.
    """

    # load_params
    base_directory = '../simulations_v2/params/baseline_testing/uw_groups/nominal/multi-group-groups/'
    ug_dorm_params = load_params(base_directory + 'ug_dorm.yaml')
    ug_off_campus_params = load_params(base_directory + 'ug_off_campus.yaml')
    gs_research_params = load_params(base_directory + 'grad_research.yaml')
    gs_other_params = load_params(base_directory + 'grad_other.yaml')
    faculty_staff_student_params = load_params(base_directory + 'faculty_staff_student_same_age.yaml')
    faculty_staff_non_student_params = load_params(base_directory + 'faculty_staff_non_student_same_age.yaml')
    faculty_staff_off_campus_params = load_params(base_directory + 'faculty_staff_off_campus_same_age.yaml')
    ithaca_community_params = load_params(base_directory + 'madison_community.yaml')
    
    # list of group parameters (single instance)
    group_params = [ug_dorm_params.copy(), ug_off_campus_params.copy(), gs_research_params.copy(), gs_other_params.copy(), faculty_staff_student_params.copy(), faculty_staff_non_student_params.copy(), faculty_staff_off_campus_params.copy(), ithaca_community_params.copy()]

    # overarching parameters - nominal, optimistic, pessimistic
    scenarios = create_scenario_dict(args)

    # translate args to iterable parameter values - input from .txt file
    dynamic_scn_params = ready_params(args)

    # matrix of lambda interactions between individuals of each group each day (should be passed in)
    # interaction_matrix =
    interaction_matrix = np.array([[12.5,  4,     0.1,   0.1,   1,     0.05,  0.05, 0.1],
                                   [3.41,  8,     0.1,   0.1,   1,     0.05,  0.05, 0.2],
                                   [0.19,  0.22,  4,     0.1,   1.2,   0.05,  0.2,  1.8],
                                   [0.14,  0.17,  0.07,  9,     1,    0.05,   0.05, 0.2],
                                   [1.92,  2.26,  1.22,  1.37,  1,     0.15,  0.3,  1.56],
                                   [0.18,  0.21,  0.1,   0.13,  0.28,  1.8,   0.2,  1.56],
                                   [0.07,  0.09,  0.15,  0.05,  0.23,  0.08,  1.8,  1.56],
                                   [0.011, 0.026, 0.106, 0.016, 0.091, 0.048, 0.12, 3.5]])

    # uw testing policy - see https://docs.google.com/document/d/1dPm0BvstSPbDc5gnpYDx6c-aUaySSq62IjIrH_rmbhw/edit
    test_fraction = [0.07142857143, 0.002945022323, 0.002987018418, 0.002987018418, 0.002987018418, 0.002987018418, 0, 0]

    group_sizes = []
    group_names = []
    for group in group_params:
        group_sizes.append(group['params']['population_size'])
        group_names.append(group['params']['group_name'])
        # group_sizes = [6931, 24254, 13012, 1120, 1698, 8083, 8084, 258054]

    # rescale interaction matrix based on group sizes
    for i in range(len(group_sizes)):
        for j in range(len(group_sizes)):
            interaction_matrix[j, i] = (interaction_matrix[i, j] * group_sizes[i]) / group_sizes[j]

    run_simulations(scenarios, int(args.ntrajectories), args.time_horizon,
                    dynamic_scn_params, interaction_matrix, group_params,
                    group_names, group_sizes, test_fraction, args)


def run_simulations(scenarios, ntrajectories, time_horizon, dynamic_scn_params,
                    interaction_matrix, group_params, group_names,
                    group_sizes, test_fraction, args):
    """
    Function to prep and submit individual simulations to a dask cluster, then
    process the results of the simulation.
    """

    # TODO: need to iterate over all of the group_params and make an iterable
    # list of group x parameter values over which to run the simulations
    # (while varying one parameter at a time) - utilizing iter_param_variations()?

    print('{}: submitting jobs...'.format(time.ctime()))

    # initialize counter
    job_counter = 0

    # collect results in array (just so we know when everything is done)
    result_collection = []

    with get_client() as client:

        for scn_name, static_scn_params in scenarios.items():

            # create directories for each scenario name

            for group_params_instance in iter_param_variations(static_scn_params, dynamic_scn_params, group_params, client):

                # submit the simulation to dask
                submit_simulation(ntrajectories,
                                  time_horizon, result_collection,
                                  interaction_matrix, group_sizes,
                                  test_fraction, group_names,
                                  group_params_instance, client)

                # keep track of how many jobs were submitted
                job_counter += 1

        process_results(result_collection, job_counter, args)


def submit_simulation(ntrajectories, time_horizon,
                      result_collection, interaction_matrix, group_sizes,
                      test_fraction, group_names, group_params, client):
    """
    Prepares a scenario for multiple iterations, submits that process to the
    dask client, and then appends the result (promise/future) to the
    result_collection
    """

    # package up inputs for running simulations
    # fn_args = (sim_sub_dir, sim_params, ntrajectories, time_horizon)

    args_for_multigroup = (group_params, interaction_matrix, group_names,
                           test_fraction, ntrajectories, time_horizon)

    # run single group simulation
    # result_collection.append(client.submit(run_background_sim, fn_args))
    
    # simulate_multiple_groups(args_for_multigroup)
    # run multi-group simulations
    result_collection.append(client.submit(simulate_multiple_groups, args_for_multigroup))


def simulate_multiple_groups(args_for_multigroup):
    """
    Was 'evaluate_testing_policy() in UW-8-group-simulations.ipynb'. Now takes
    a tuple of arguments for the simulation and executes the multi-group
    simulation, returning a dataframe containing the results of the simulation
    """
    group_params = args_for_multigroup[0]
    interaction_matrix = args_for_multigroup[1]
    group_names = args_for_multigroup[2]
    test_frac = args_for_multigroup[3]
    ntrajectories = args_for_multigroup[4]
    time_horizon = args_for_multigroup[5]

    static_group_params = []
    for group in group_params:
        static_group_params.append(group['params'])

    assert len(group_params) == len(test_frac)

    group_size = list()
    tests_per_day = 0

    # set group based contacts per day, test frequency
    for index, params in enumerate(static_group_params):
        params['expected_contacts_per_day'] = interaction_matrix[index, index]
        params['test_population_fraction'] = test_frac[index]
        group_size.append(params['population_size'])
        tests_per_day += group_size[-1] * test_frac[index]

    assert len(group_size) == len(test_frac)

    sim = MultiGroupSimulation(static_group_params, interaction_matrix, group_names)
    sim_results = run_multigroup_multiple_trajectories(sim, time_horizon, ntrajectories)
    return interaction_matrix, test_frac, ntrajectories, time_horizon, group_names, static_group_params, sim_results


def run_multigroup_sim(sim, T):
    sim.run_new_trajectory(T)
    list_dfs = list()
    for sim_group in sim.sims:
        list_dfs.append(sim_group.sim_df)
    return list_dfs


def run_multigroup_multiple_trajectories(sim, T, n):
    sim_results = list()
    for _ in range(n):
        result = run_multigroup_sim(sim, T)
        sim_results.append(result)
    return sim_results


def run_background_sim(input_tuple):
    """
    Main process passed to dask client to run multiple replicates of a single
    instance (unique parameter set) of the simulation.
    """

    output_dir = input_tuple[0]
    sim_params = input_tuple[1]
    ntrajectories = input_tuple[2]
    time_horizon = input_tuple[3]

    dfs = run_multiple_trajectories(sim_params, ntrajectories, time_horizon)

    return output_dir, dfs


def process_results(result_collection, job_counter, args):
    """
    Takes the collection of futures returned from the dask process and
    iterates over all of them, writing the results to files as they are
    returned.
    """

    # counter to iterate over and process all results
    get_counter = 0

    engine = sqlalchemy.create_engine(db_config.config_string)

    for result in result_collection:

        # make sure we are getting both input and output back!

        # pool approach
        # result.get()

        # dask approach
        output = result.result()

        sim_id = uuid.uuid4()

        # write sim params
        interaction_matrix = output[0]
        test_frac = output[1]
        ntrajectories = output[2]
        time_horizon = output[3]
        group_names = output[4]

        sim_params = pd.DataFrame({
            'test_frac': test_frac,
            'ntrajectories': ntrajectories,
            'time_horizon': time_horizon,
            'group_names': group_names,
            'sim_id': sim_id
            })

        sim_params['contact_rates'] = interaction_matrix.tolist()

        sim_params.to_sql('sim_params', con=engine, index_label='group_index', if_exists='append', method='multi')

        # sim_params = pd.DataFrame()
        # sim_params.at[0, 'ntrajectories'] = ntrajectories
        # sim_params.at[0, 'time_horizon'] = time_horizon
        #
        # sim_params['test_frac'] = None
        # sim_params['test_frac'] = sim_params['test_frac'].astype(object)
        # sim_params.at[0, 'test_frac'] = test_frac
        #
        # sim_params['interaction_matrix'] = None
        # sim_params['interaction_matrix'] = sim_params['interaction_matrix'].astype(object)
        # sim_params['interaction_matrix'] = interaction_matrix.tolist()
        #
        # sim_params = pd.DataFrame({
        #     'interaction_matrix': interaction_matrix,
        #     'test_frac': test_frac,
        #     'ntrajectories': ntrajectories,
        #     'time_horizon': time_horizon
        #     })
        #
        # sim_params = pd.DataFrame({'test_frac': test_frac, 'ntrajectories': ntrajectories, 'time_horizon': time_horizon })

        # write group params
        for group_number in range(len(output[5])):
            output[5][group_number]['sim_id'] = sim_id
            param_df = pd.DataFrame(output[5][group_number]).iloc[[0], 1:]
            param_df['group_number'] = group_number

            # stupid hacky stuff to get it to store an array in a cell and write it to the db
            param_df['severity_prevalence'] = None
            param_df.at[0, 'severity_prevalence'] = output[5][group_number]['severity_prevalence'].tolist()

            # write to database
            param_df.to_sql('group_params', con=engine, if_exists='append', method='multi')

        # write results
        for trajectory_number in range(len(output[6])):
            replicate_id = uuid.uuid4()
            for group_number in range(len(output[6][trajectory_number])):
                output[6][trajectory_number][group_number]['sim_id'] = sim_id
                output[6][trajectory_number][group_number]['replicate_id'] = replicate_id
                output[6][trajectory_number][group_number].to_sql('results', con=engine, if_exists='append', method='multi')

        get_counter += 1

        print("{}: {} of {} simulations complete!".format(time.ctime(), get_counter, job_counter))

    print("Simulations done.")


def initialize_results_directory(sim_scn_dir, job_counter, param_specifier,
                                 sim_params, args):
    """
    Creates the required directory structure to store the simulation results
    along with the intial simulation yaml files.
    """

    sim_sub_dir = "{}/simulation-{}".format(sim_scn_dir, job_counter)
    os.mkdir(sim_sub_dir)

    with open('{}/param_specifier.yaml'.format(sim_sub_dir), 'w') as outfile:
        yaml.dump(param_specifier, outfile, default_flow_style=False)
        if args.verbose:
            print("Created directory {} to save output".format(sim_sub_dir))

        dill.dump(sim_params, open("{}/sim_params.dill".format(sim_sub_dir), "wb"))

    return sim_sub_dir


def get_client():
    if socket.gethostname() == 'submit3.chtc.wisc.edu':
        # CHTC execution
        from dask.distributed import Worker, WorkerPlugin
        from dask_chtc import CHTCCluster
        from typing import List

        '''
        # method to import libraries to workers - from https://github.com/dask/distributed/issues/1200#issuecomment-653495399
        class DependencyInstaller(WorkerPlugin):
            def __init__(self, dependencies: List[str]):
                self._depencendies = " ".join(f"'{dep}'" for dep in dependencies)

            def setup(self, _worker: Worker):
                os.system(f"pip install {self._depencendies}")

        dependency_installer = DependencyInstaller([
            "scipy",
            "functools",
            "numpy",
            "pandas"
        ])
        '''

        cluster = CHTCCluster(worker_image="blue442/group-modeling-chtc:0.1", job_extra={"accounting_group": "COVID19_AFIDSI"})
        cluster.adapt(minimum=10, maximum=20)
        client = Client(cluster)

        # install packages to client
        # client.register_worker_plugin(dependency_installer)
        # Does this work???
        # client.upload_file('analysis_helpers.py')
        # client.upload_file('stochastic_simulation.py')
    else:
        # local execution
        cluster = LocalCluster(multiprocessing.cpu_count() - 1)
        client = Client(cluster)
    return client


def ready_params(args):
    """
    Uses the desired variation in parameters to create a dictionary of
    simulation parameters to be used in a single instance of the simulation.
    """

    param_values = {}
    params_to_vary = args.param_to_vary

    for param_to_vary, values in zip(params_to_vary, args.values):
        if param_to_vary not in VALID_GLOBAL_PARAMS_TO_VARY:
            print("Received invalid parameter to vary: {}".format(param_to_vary))
            exit()
        if param_to_vary == 'contact_tracing_delay':
            param_values[param_to_vary] = [int(v) for v in values]
        else:
            param_values[param_to_vary] = [float(v) for v in values]

    return param_values


def create_directories(args):
    """
    Utilize timestamps to create unique 'names' for simulation runs, and create
    a parent directory (using that name) under which all simulations from this
    scenario are stored.
    """

    params_to_vary = args.param_to_vary

    if len(params_to_vary) == 1:
        sim_id = "{timestamp}-{param_to_vary}".format(
                    timestamp=str(time.time()),
                    param_to_vary=params_to_vary[0])
    else:
        sim_id = "{timestamp}-multiparam".format(
                    timestamp=str(time.time()).split('.')[0])

    print("Using Simulation ID: {}".format(sim_id))

    basedir = args.outputdir

    if not os.path.isdir(basedir):
        print("Directory {} does not exist. Please create it.".format(basedir))
        exit()

    sim_main_dir = basedir + "/" + str(sim_id)
    os.mkdir(sim_main_dir)
    print("Output directory {} created".format(sim_main_dir))

    return sim_main_dir


# def iter_param_variations(base_params, params_to_vary, param_values):
def iter_param_variations(static_scn_params, dynamic_scn_params, group_params, client):
    """
    iterator that generates all parameter configurations corresponding to
    all combinations of parameter values across the different params_to_vary.
    Each return value is a tuple (param_specifier, params) where params is the
    parameter dictionary object, and param_specifier is a smaller dict
    specifying the varying params and the value they are taking right now
    """

    # group_params_original = copy.deep_copy(group_params)

    # iterate over the global scenario parameters to vary
    for p_name, p_values in dynamic_scn_params.items():
        print(p_name, p_values)

    for group in group_params:
        for param_to_vary, param_vals in group['dynamic_params'].items():
            for param_val in param_vals:
                # returnable_group_params = copy.deep_copy(group_params_original)
                update_params(group['params'], param_to_vary, param_val)   # this will update group['params'] to have one instance of the dynamic values
                yield group_params

    """
    # older approach (single group)
    base_params = base_params.copy()
    params_list = [param_values[param] for param in params_to_vary]
    for param_tuple in itertools.product(*params_list):
        param_specifier = {}
        for param, value in zip(params_to_vary, param_tuple):
            update_params(base_params, param, value)
            param_specifier[param] = value

        yield param_specifier, base_params
    """

def update_params(sim_params, param_to_vary, param_val):
    # VERY TEMPORARY HACK TO GET SENSITIVITY SIMS FOR ASYMPTOMATIC %
    if param_to_vary == 'asymptomatic_p':
        assert(sim_params['mild_severity_levels'] == 1)
        curr_prevalence_dist = sim_params['severity_prevalence']
        assert(param_val >= 0 and param_val <= 1)
        new_dist = [param_val]
        remaining_mass = sum(curr_prevalence_dist[1:])

        # need to scale so that param_val + x * remaning_mass == 1
        scale = (1 - param_val) / remaining_mass
        idx = 1
        while idx < len(curr_prevalence_dist):
            new_dist.append(curr_prevalence_dist[idx] * scale)
            idx += 1
        assert(np.isclose(sum(new_dist), 1))
        sim_params['severity_prevalence'] = np.array(new_dist)

    elif param_to_vary == 'symptomatic_daily_self_report_p':
        sim_params['severe_symptoms_daily_self_report_p'] = param_val
    elif param_to_vary == 'asymptomatic_daily_self_report_p':
        sim_params['mild_symptoms_daily_self_report_p'] = param_val

    # VERY TEMPORARY HACK TO GET SENSITIVITY SIMS WORKING FOR CONTACT RECALL %
    elif param_to_vary == 'contact_tracing_constant':
        num_isolations = sim_params['cases_isolated_per_contact']
        base_recall = sim_params['contact_recall']
        new_isolations = num_isolations * param_val / base_recall
        new_quarantines = max(7 - new_isolations, 0)
        sim_params['cases_isolated_per_contact'] = new_isolations
        sim_params['cases_quarantined_per_contact'] = new_quarantines
    elif param_to_vary == 'contact_tracing_isolations':
        sim_params['cases_isolated_per_contact'] = param_val
        sim_params['cases_quarantined_per_contact'] = 7 - param_val
    else:
        sim_params[param_to_vary] = param_val


#################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run multiple simulations using multiprocessing')
    parser.add_argument('-o', '--outputdir', default=BASE_DIRECTORY,
                        help='directory to store simulation output')
    parser.add_argument('-V', '--verbose', action='store_true', help='include verbose output')
    parser.add_argument('-s', '--scenarios', nargs='+', required=True,
                        help='list of YAML config files specifying base sets of scenario parameters to use')

    parser.add_argument('-p', '--param-to-vary', action='append',
                        help='which param(s) should be varied in the corresponding sensitivity sims', required=True)

    parser.add_argument('-v', '--values', required=True, nargs='+', action='append',
                        help='what values should the varying parameter(s) take')

    parser.add_argument('-n', '--ntrajectories', default=500,
                        help='how many trajectories to simulate for each (scenario, value) pair')
    parser.add_argument('-t', '--time-horizon', default=112,
                        help='how many days to simulate for each trajectory')

    parser.add_argument('-f', '--fig-dir',
                        help='specify folder where plots should be saved')

    args = parser.parse_args()

    # multithreading process method
    # old_simulate(args)

    # multithreading.pool method
    simulate(args)
