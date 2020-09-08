import argparse
import itertools
import multiprocessing
import os
import pdb
import socket
import sys
import time
import yaml

import dill
from dask.distributed import Client, LocalCluster
from dask_chtc import CHTCCluster
import numpy as np

from analysis_helpers import run_multiple_trajectories
from load_params import load_params
from multi_group_simulation import MultiGroupSimulation
from plotting_util import plot_from_folder

BASE_DIRECTORY = os.path.abspath(os.path.join('')) + "/sim_output/"

VALID_PARAMS_TO_VARY = [
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

    scenarios = create_scenario_dict(args)

    param_values = ready_params(args)

    sim_main_dir = create_directories(args)

    run_simulations(scenarios, int(args.ntrajectories), args.time_horizon,
                    param_values, sim_main_dir, args)


def run_simulations(scenarios, ntrajectories, time_horizon, param_values,
                    sim_main_dir, args):
    """
    Function to prep and submit individual simulations to a dask cluster, then
    process the results of the simulation.
    """

    print('{}: submitting jobs...'.format(time.ctime()))

    params_to_vary = args.param_to_vary

    # initialize counter
    job_counter = 0

    # collect results in array (just so we know when everything is done)
    result_collection = []

    with get_client() as client:

        for scn_name, scn_params in scenarios.items():

            # create directories for each scenario name
            sim_scn_dir = sim_main_dir + "/" + scn_name
            os.mkdir(sim_scn_dir)

            # dump params into a dill file
            dill.dump(scn_params, open("{}/scn_params.dill".format(sim_scn_dir),
                      "wb"))

            for param_specifier, sim_params in iter_param_variations(scn_params, params_to_vary, param_values, client):

                # initialize the relevant subdirectory
                sim_sub_dir = initialize_results_directory(sim_scn_dir,
                                                           job_counter,
                                                           param_specifier,
                                                           sim_params, args)

                # submit the simulation to dask
                submit_simulation(sim_params, param_specifier, ntrajectories,
                                  time_horizon, result_collection,
                                  client)

                # keep track of how many jobs were submitted
                job_counter += 1

        process_results(result_collection, job_counter, params_to_vary, args,
                        sim_main_dir)


def submit_simulation(sim_params, param_specifier, ntrajectories, time_horizon,
                      result_collection, client):
    """
    Prepares a scenario for multiple iterations, submits that process to the
    dask client, and then appends the result (promise/future) to the
    result_collection
    """

    # package up inputs for running simulations
    # fn_args = (sim_sub_dir, sim_params, ntrajectories, time_horizon)
    params_list = 
    interaction_matrix = 
    group_names = 
    test_frac = 

    args_for_multigroup = (params_list, interaction_matrix, group_names,
                           test_frac, ntrajectories, time_horizon)

    # run single group simulation
    # result_collection.append(client.submit(run_background_sim, fn_args))

    # run multi-group simulations
    result_collection.append(client.submit(simulate_multiple_groups, args_for_multigroup))


def simulate_multiple_groups(args_for_multigroup):
    """
    Was 'evaluate_testing_policy() in UW-8-group-simulations.ipynb'. Now takes
    a tuple of arguments for the simulation and executes the multi-group
    simulation, returning a dataframe containing the results of the simulation
    """

    params_list = args_for_multigroup[0]
    interaction_matrix = args_for_multigroup[1]
    group_names = args_for_multigroup[2]
    test_frac = args_for_multigroup[3]
    ntrajectories = args_for_multigroup[4]
    time_horizon = args_for_multigroup[5]

    assert len(params_list) == len(test_frac)

    group_size = list()
    tests_per_day = 0

    # set group based contacts per day, test frequency
    for index, params in enumerate(params_list):
        params['expected_contacts_per_day'] = interaction_matrix[index, index]
        params['test_population_fraction'] = test_frac[index]
        group_size.append(params['population_size'])
        tests_per_day += group_size[-1] * test_frac[index]

    assert len(group_size) == len(test_frac)

    sim = MultiGroupSimulation(params_list, interaction_matrix, group_names)
    sim_results = run_multiple_trajectories(sim, ntrajectories, time_horizon)
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


def process_results(result_collection, job_counter, params_to_vary, args,
                    sim_main_dir):
    """
    Takes the collection of futures returned from the dask process and
    iterates over all of them, writing the results to files as they are
    returned.
    """

    # counter to iterate over and process all results
    get_counter = 0

    for result in result_collection:

        # pool approach
        # result.get()

        # dask approach
        output = result.result()

        output_dir = output[0]
        dfs = output[1]

        for idx, df in enumerate(dfs):
            df_file_name = "{}/{}.csv".format(output_dir, idx)
            df.to_csv(df_file_name)

        get_counter += 1

        print("{}: {} of {} simulations complete!".format(time.ctime(), get_counter, job_counter))

    # wrap up
    if len(params_to_vary) > 1:
        print("Simulations done. Not auto-generating plots because > 1 parameter was varied")
        print("Exiting now...")
        exit()

    print("Simulations done. Generating plots now...")
    if args.fig_dir is None:
        fig_dir = sim_main_dir
    else:
        fig_dir = args.fig_dir
    plot_from_folder(sim_main_dir, fig_dir)
    print("Saved plots to directory {}".format(fig_dir))


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
    """
    Obtains a dask client which distributes the tasks (simulations) across
    multiple workers. Hardcoded to look for current hostname, which if it
    resolves to CHTC will execute in. CHTC pool, otherwise creates a client
    (and cluster) utilzing all but one CPU on the local machine.
    """

    if socket.gethostname() == 'submit3.chtc.wisc.edu':
        # CHTC execution
        cluster = CHTCCluster(job_extra={"accounting_group": "COVID19_AFIDSI"})
        cluster.adapt(minimum=10, maximum=20)
        client = Client(cluster)
        client.upload_file('analysis_helpers.py')
        client.upload_file('stochastic_simulation.py')

    else:
        # local execution
        cluster = LocalCluster(multiprocessing.cpu_count() - 1)
        client = Client(cluster)

    return client


def create_scenario_dict(args):
    """
    Load the parameter values for each scenario from file, return a dictionary
    containing each parameter set keyed by the scenario name.
    """

    scenarios = {}
    for scenario_file in args.scenarios:
        # scn_params object is a dictionary of parameters loaded into the
        # 'base_parms' object in load_params.py
        scn_name, scn_params = load_params(scenario_file)
        scenarios[scn_name] = scn_params
    return scenarios


def ready_params(args):
    """
    Uses the desired variation in parameters to create a dictionary of
    simulation parameters to be used in a single instance of the simulation.
    """

    param_values = {}
    params_to_vary = args.param_to_vary

    for param_to_vary, values in zip(params_to_vary, args.values):
        if param_to_vary not in VALID_PARAMS_TO_VARY:
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


def iter_param_variations(base_params, params_to_vary, param_values):
    """
    iterator that generates all parameter configurations corresponding to
    all combinations of parameter values across the different params_to_vary.
    Each return value is a tuple (param_specifier, params) where params is the
    parameter dictionary object, and param_specifier is a smaller dict
    specifying the varying params and the value they are taking right now
    """

    base_params = base_params.copy()
    params_list = [param_values[param] for param in params_to_vary]
    for param_tuple in itertools.product(*params_list):
        param_specifier = {}
        for param, value in zip(params_to_vary, param_tuple):
            update_params(base_params, param, value)
            param_specifier[param] = value

        yield param_specifier, base_params


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


def load_multigroup_params(group_yaml_list):
    """
    Takes a collection of strings representing names of yaml files used to 
    parameterize the different groups and loads them all into a different
    collection containing the intantiation of those yaml files
    """

    params_list = []
    for yaml_file in group_yaml_list:
        params = load_params(yaml_file)
        params_list.append(params.copy())

    return params_list


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
