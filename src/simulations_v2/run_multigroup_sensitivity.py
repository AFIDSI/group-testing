import argparse
import copy
import itertools
import logging
from logging.config import fileConfig
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
# from multi_group_simulation import MultiGroupSimulation
from load_multigroup_params import load_params, load_parameters_from_yaml, identify_dynamic_params
from logger_initializer import initialize_logger
from run_sensitivity import create_scenario_dict

# fileConfig('logging.ini')
# logger = logging.getLogger(__name__)
# logger.addHandler('file_logger')
initialize_logger('./logs')

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

BATCH_SIZE = 100

def simulate(args):
    """
    Main function that initializes the simulations, executes them, and then
    manages the results.
    """

    group_sizes = []

    params = load_parameters_from_yaml(args.multigroup_file)

    dynamic_params = identify_dynamic_params(params)

    # create list of permutations
    keys, values = zip(*dynamic_params.items())
    dynamic_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    group_params = params['groups']
    interaction_matrix = params['interaction_matrix']

    # rescale interaction matrix based on group sizes
    for i in range(len(group_sizes)):
        for j in range(len(group_sizes)):
            interaction_matrix[j, i] = (interaction_matrix[i, j] * group_sizes[i]) / group_sizes[j]

    run_simulations(params['ntrajectories'], params['time_horizon'],
                    dynamic_permutations, interaction_matrix, group_params,
                    group_sizes, BATCH_SIZE, args)


def run_simulations(ntrajectories, time_horizon, dynamic_permutations,
                    interaction_matrix, group_params,
                    group_sizes, batch_size, args):
    """
    Function to prep and submit individual simulations to a dask cluster, then
    process the results of the simulation.
    """
    submit_time = time.ctime()
    logging.info('{}: submitting jobs...'.format(submit_time))

    # initialize counter
    job_counter = 0

    # collect results in array (just so we know when everything is done)
    result_collection = []

    with get_client() as client:

        for group_params_instance in iter_param_variations(dynamic_permutations, group_params):

            if len(result_collection) < batch_size:

                # create unique id for simulation
                sim_id = uuid.uuid4()

                # submit the simulation to dask
                submit_simulation(ntrajectories,
                                  time_horizon, result_collection,
                                  interaction_matrix, group_sizes,
                                  group_params_instance, client, sim_id,
                                  job_counter)

            else:

                process_results(result_collection, job_counter, args, submit_time)
                result_collection = []

        # catch the last batch
        process_results(result_collection, job_counter, args, submit_time)

    logging.info("Processing of simulations complete!")


def submit_simulation(ntrajectories, time_horizon,
                      result_collection, interaction_matrix, group_sizes,
                      group_params, client, sim_id, job_counter):
    """
    Prepares a scenario for multiple iterations, submits that process to the
    dask client, and then appends the result (promise/future) to the
    result_collection
    """

    # package up inputs for running simulations
    # fn_args = (sim_sub_dir, sim_params, ntrajectories, time_horizon)

    args_for_multigroup = (group_params, interaction_matrix,
                           ntrajectories, time_horizon)

    # run single group simulation
    # result_collection.append(client.submit(run_background_sim, fn_args))

    # simulate_multiple_groups(args_for_multigroup)
    # run multi-group simulations
    sim = initialize_multigroup_sim(args_for_multigroup)

    # result_collection.append(client.submit(simulate_multiple_groups, args_for_multigroup))
    for _ in range(ntrajectories):
        replicate_id = uuid.uuid4()
        # keep track of how many jobs were submitted
        job_counter += 1
        submittable_sim = copy.deepcopy(sim)
        submittable_sim.sim_id = sim_id
        submittable_sim.replicate_id = replicate_id
        logging.info('submitting sim_id = {}, replicate_id = {}'.format(submittable_sim.sim_id, submittable_sim.replicate_id))

        result_collection.append(client.submit(submittable_sim.run_multigroup_sim))


def initialize_multigroup_sim(args_for_multigroup):
    """
    Was 'evaluate_testing_policy() in UW-8-group-simulations.ipynb'. Now takes
    a tuple of arguments for the simulation and executes the multi-group
    simulation, returning a dataframe containing the results of the simulation
    """

    # FIXME: expecting args_for_multigroup[0] to be a list of parameter dictionaries for each group ordered by index
    # instead, it's a dictionary of params keyed by group name

    group_params = args_for_multigroup[0]
    interaction_matrix = args_for_multigroup[1]
    ntrajectories = args_for_multigroup[2]
    time_horizon = args_for_multigroup[3]

    static_group_params = []
    group_names = []
    for group_name, group_params in group_params.items():
        static_group_params.append(group_params)
        group_names.append(group_name)

    group_size = list()
    tests_per_day = 0

    # set group based contacts per day from interaction matrix
    for index, params in enumerate(static_group_params):
        params['expected_contacts_per_day'] = interaction_matrix[index, index]
        group_size.append(params['population_size'])
        tests_per_day += group_size[-1] * params['test_population_fraction']

    sim = MultiGroupSimulation(static_group_params, interaction_matrix, ntrajectories, time_horizon, group_names)

    return sim


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


def process_results(result_collection, job_counter, args, submit_time):
    """
    Takes the collection of futures returned from the dask process and
    iterates over all of them, writing the results to files as they are
    returned.

    MultigroupSimulation.run_multigroup_sim() returns: self.interaction_matrix, self.ntrajectories, self.time_horizon, self.group_names, self.group_params, group_results
    """

    # counter to iterate over and process all results
    get_counter = 0

    engine = sqlalchemy.create_engine(db_config.config_string)

    for result in result_collection:
        logging.debug('waiting for results')
        output = result.result()
        logging.debug('writing result to database')
        result_to_database(output, engine, submit_time, job_counter, get_counter)

    logging.info('batch written to database')


def result_to_database(output, engine, submit_time, job_counter, get_counter):

    # write sim params
    interaction_matrix = output[0]
    ntrajectories = output[1]
    time_horizon = output[2]
    group_names = output[3]
    sim_id = output[6]
    replicate_id = output[7]

    sim_params = pd.DataFrame({
        'ntrajectories': ntrajectories,
        'time_horizon': time_horizon,
        'group_names': group_names,
        'sim_id': sim_id
        })

    sim_params['contact_rates'] = interaction_matrix.tolist()

    sim_params.to_sql('sim_params', con=engine, index_label='group_index', if_exists='append', method='multi')

    # write group params
    for group_number in range(len(output[4])):
        output[4][group_number]['sim_id'] = sim_id

        # remove non-conforming entries
        severity_prevalence = output[4][group_number].pop('severity_prevalence').tolist()
        age_distribution = output[4][group_number].pop('age_distribution')

        # param_df = pd.DataFrame(output[4][group_number]).iloc[[0], 1:]
        param_df = pd.DataFrame(output[4][group_number], index=[0])
        param_df['group_number'] = group_number

        # stupid hacky stuff to get it to store an array in a cell and write it to the db
        param_df['severity_prevalence'] = None
        param_df['age_distribution'] = None

        # param_df.at[0, 'severity_prevalence'] = output[4][group_number]['severity_prevalence'].tolist()
        param_df.at[0, 'severity_prevalence'] = severity_prevalence
        param_df.at[0, 'age_distribution'] = age_distribution
        param_df.at[0, 'submit_time'] = submit_time

        # write to database
        param_df.to_sql('group_params', con=engine, if_exists='append', method='multi')

    # write results
    for group_number in range(len(output[5])):
        output[5][group_number]['sim_id'] = sim_id
        output[5][group_number]['replicate_id'] = replicate_id
        output[5][group_number]['group_number'] = group_number
        output[5][group_number].to_sql('results', index_label='t', con=engine, if_exists='append', method='multi')

    get_counter += 1

    logging.info("{}: {} of {} simulations complete!".format(time.ctime(), get_counter, job_counter))


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

        cluster = CHTCCluster(worker_image="blue442/group-modeling-chtc:0.1", job_extra={"accounting_group": "COVID19_AFIDSI"})
        cluster.adapt(minimum=BATCH_SIZE, maximum=BATCH_SIZE)
        client = Client(cluster)

    else:
        # local execution
        cluster = LocalCluster(multiprocessing.cpu_count() - 1)
        client = Client(cluster)
    logging.info('CLIENT SERVICES: {}'.format(client.scheduler_info()['services']))
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
            logging.error("Received invalid parameter to vary: {}".format(param_to_vary))
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
def iter_param_variations(dynamic_scn_params, params):
    """
    iterator that generates all parameter configurations corresponding to
    all combinations of parameter values across the different params_to_vary.
    Each return value is a tuple (param_specifier, params) where params is the
    parameter dictionary object, and param_specifier is a smaller dict
    specifying the varying params and the value they are taking right now
    """
    # pdb.set_trace()
    # iterate over the global scenario parameters to vary
    i = 0
    for iteration in dynamic_scn_params:
        i += 1
        logging.debug('parameter set #{}'.format(i))
        for parameter_assignment, parameter_value in iteration.items():
            logging.debug('     group: {}, parameter name: {}, parameter value: {}'.format(parameter_assignment[0], parameter_assignment[1], parameter_value))

    for permutation in dynamic_scn_params:

        # create a copy of the original
        permuted_group_params = copy.deepcopy(params)

        for parameter_assignment, parameter_value in permutation.items():

            # permute the copy of the original to have one instance of the dynamic values
            update_params(permuted_group_params[parameter_assignment[0]], parameter_assignment[1], parameter_value)

        # yield the permuted set of parameters to be run
        yield permuted_group_params

    '''
    for group_name in group_params.keys():
        for param_to_vary, param_vals in group_params[group_number]['dynamic_params'].items():
            for param_val in param_vals:

                # create a copy of the original
                permuted_group_params = copy.deepcopy(params)

                # permute the copy of the original to have one instance of the dynamic values
                update_params(permuted_group_params[group_name], param_to_vary, param_val)

                # yield the permuted set of parameters to be run
                yield permuted_group_params
    '''

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


################
# Define classes locally to avoid import errors in workers
################

import numpy as np

class MultiGroupSimulation:
    def __init__(self,
                 group_params,
                 interaction_matrix,
                 ntrajectories,
                 time_horizon,
                 group_names=[]):
        """
        group_params: A list of dictionaries of length N.  Each dictionary is
                    used as the config for a different individual-group
                    StochasticSimulation object.
        interaction_matrix: A N x N matrix such that the (i,j) element indicates
                    the rate at which members of group i are exposed to members of
                    group j.
                    Specifically, each free member of group i has Poisson(lambda[i,j])
                    contacts each day with a free member of group j
        group_names: An optional list of strings of length N indicating the name
                    of each group
        """
        self.sims = [CHTCStochasticSimulation(params) for params in group_params]
        self.N = len(group_params)
        self.interaction_matrix = interaction_matrix
        self.ntrajectories = ntrajectories
        self.time_horizon = time_horizon
        self.group_names = group_names
        self.group_params = group_params

        self.original_interaction_matrix = interaction_matrix
        self.original_daily_contacts = [sim.daily_contacts_lambda for sim in self.sims]
        self.lockdown_in_effect = False
        self.simulate_lockdown = False
        self.sim_id = None
        self.replicate_id = None

    def configure_lockdown(self,
                           post_lockdown_interaction_matrix,
                           new_case_sims_list,
                           new_cases_threshold,  # observed new cases found by testing / self-reporting that trigger lockdown
                                                 # specified as raw number of cases
                           new_cases_time_window,  # number of days over which new cases are computed for the previous threshold
                           use_second_derivative=False,
                           second_derivative_threshold=None
                           ):
        self.simulate_lockdown = True
        self.new_case_sims_list = new_case_sims_list
        assert(len(self.new_case_sims_list) == len(self.sims))
        self.post_lockdown_interaction_matrix = post_lockdown_interaction_matrix
        self.new_cases_threshold = new_cases_threshold
        self.new_cases_time_window = new_cases_time_window
        self.new_case_counts = [0] * new_cases_time_window * 2

        self.use_second_derivative = use_second_derivative
        self.second_deriv_threshold = second_derivative_threshold

    def step_lockdown_status(self, t):
        assert(self.simulate_lockdown)
        self.update_case_counts()
        new_cases = self.get_new_case_counts()
        second_deriv = self.second_derivative_estimate()
        # Note - changing new_cases_threshold to raw number rather than a proportion
        if new_cases >= self.new_cases_threshold or \
                (t >= self.new_cases_time_window * 2 and self.use_second_derivative and second_deriv >= self.second_deriv_threshold):
            self.lockdown_in_effect = True
            self.interaction_matrix = self.post_lockdown_interaction_matrix
            for i in range(self.N):
                self.sims[i].daily_contacts_lambda = self.post_lockdown_interaction_matrix[i,i]

    def get_new_case_counts(self):
        return sum(self.new_case_counts[self.new_cases_time_window:2*self.new_cases_time_window])

    def second_derivative_estimate(self):
        # new_case_counts ~ Y(t) - Y(t-window)
        # so derivative = new_case_counts / window
        first_deriv = self.get_new_case_counts() / self.new_cases_time_window

        prev_first_deriv = sum(self.new_case_counts[0:self.new_cases_time_window]) / self.new_cases_time_window

        return (first_deriv - prev_first_deriv) / self.new_cases_time_window

    def update_case_counts(self):
        new_cases_today = 0
        for sim, include in zip(self.sims, self.new_case_sims_list):
            if include:
                new_cases_today += sim.new_QS_from_last_test
                new_cases_today += sim.new_QI_from_last_test
                new_cases_today += sim.new_QI_from_self_reports

        #shift case count array down
        self.new_case_counts.pop(0)
        self.new_case_counts.append(new_cases_today)

    def get_interaction_mtx(self):
        return self.interaction_matrix

    def get_total_population(self):
        return sum([sim.pop_size for sim in self.sims])

    def set_interaction_mtx(self, interaction_mtx):
        self.interaction_matrix = interaction_mtx

    def reset_initial_state(self):
        self.lockdown_in_effect = False
        self.interaction_matrix = self.original_interaction_matrix

        if self.simulate_lockdown:
            self.new_case_counts = [0] * self.new_cases_time_window * 2
            for sim, contacts in zip(self.sims, self.original_daily_contacts):
                sim.daily_contacts_lambda = contacts

        for sim in self.sims:
            sim.reset_initial_state()

    def run_new_trajectory(self, T):
        self.reset_initial_state()
        lockdown_statuses = []
        for t in range(T):
            self.step()
            if self.simulate_lockdown:
                self.step_lockdown_status(t)
            lockdown_statuses.append(self.lockdown_in_effect)

        for sim in self.sims:
            sim.update_severity_levels()

        sim_df = self.sims[0].sim_df
        for sim in self.sims[1:]:
            sim_df = sim_df.add(sim.sim_df)
        return lockdown_statuses, sim_df

    def get_free_total(self, i):
        # get the free-total count from group i
        free_total = self.get_free_infectious(i)

        if self.sims[i].pre_ID_state == 'detectable':
            free_total += sum(self.sims[i].pre_ID)

        free_total += self.sims[i].S + self.sims[i].R + sum(self.sims[i].E)
        return free_total

    def get_free_infectious(self, i):
        # get the free-infectious total from group j

        if self.sims[i].pre_ID_state == 'infectious':
            free_infectious = sum(self.sims[i].pre_ID)
        else:
            free_infectious = 0

        free_infectious += sum(self.sims[i].ID)
        free_infectious += sum(self.sims[i].SyID_mild)
        free_infectious += sum(self.sims[i].SyID_severe)

        return free_infectious

    def get_quarantine_susceptible(self, i):
        return self.sims[i].QS

    def get_quarantine_infected(self, i):
        return self.sims[i].QI

    def step(self):
        # do inter-group interactions first, so that no updates happen after each sim adds
        # a row to their dataframe
        new_E_holder = [0] * self.N
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                free_susceptible_i = self.sims[i].S

                interactions_lambda_i_j = self.interaction_matrix[i,j] * self.sims[i].contact_rate_multiplier

                free_infectious_j = self.get_free_infectious(j)
                free_total_j = self.get_free_total(j)

                poisson_param = free_susceptible_i * interactions_lambda_i_j * \
                    free_infectious_j / free_total_j

                n_susceptible_infectious_contacts = np.random.poisson(poisson_param)

                new_E = np.random.binomial(n_susceptible_infectious_contacts, self.sims[i].exposed_infection_p)
                new_E_holder[i] += new_E

        for i in range(self.N):
            self.sims[i].add_new_infections(new_E_holder[i])

        # do individual-group steps
        for sim in self.sims:
            sim.step()

    def run_multigroup_sim(self):
        self.run_new_trajectory(self.time_horizon)
        group_results = list()
        for sim_group in self.sims:
            group_results.append(sim_group.sim_df)
        return self.interaction_matrix, self.ntrajectories, self.time_horizon, self.group_names, self.group_params, group_results, self.sim_id, self.replicate_id


import numpy as np
import pandas as pd
from math import ceil
from scipy.stats import poisson
# import functools


class CHTCStochasticSimulation:

    # @functools.lru_cache(maxsize=128)
    def poisson_pmf(self, max_time, mean_time):
        pmf = list()
        for i in range(max_time):
            pmf.append(poisson.pmf(i, mean_time))
        pmf.append(1-np.sum(pmf))
        return np.array(pmf)

    def binomial_exit_function(self, n, p):
        # return (lambda n: np.random.binomial(n, p))
        return np.random.binomial(n, p)

    def poisson_waiting_function2(self, n, max_time, mean_time):
        return np.random.multinomial(n, self.poisson_pmf(max_time, mean_time))

    def multinomial_sample(self, n, max_time, mean_time):
        return np.random.multinomial(n, self.poisson_pmf(max_time, mean_time))

    def __init__(self, params):

        self.params = params

        # Meta-parameters governing the maximum number of days an
        # individual spends in each 'infection' state
        if 'max_time_E' in params:
            self.max_time_E = params['max_time_E']
        else:
            self.max_time_E = params['max_time_exposed']
        self.max_time_pre_ID = params['max_time_pre_ID']
        self.max_time_ID = params['max_time_ID']
        self.max_time_SyID_mild = params['max_time_SyID_mild']
        self.max_time_SyID_severe = params['max_time_SyID_severe']

        self.mean_time_ID = params['mean_time_ID']
        self.mean_time_SyID_mild = params['mean_time_SyID_mild']
        self.mean_time_SyID_severe = params['mean_time_SyID_severe']

        # parameters governing distribution over time spent in each
        # of these infection states:
        # Assumptions about the sample_X_times variables:
        # sample_X_times(n) returns a numpy array times of length max_time_X+1
        # such that times[k] is the number of people who stay in state X
        # for k time periods.
        # (However, the length of the corresponding queue arrays will just be max_time_X,
        # not max_time_X + 1)
        # We assume that sum(times) == n

        if 'mean_time_E' in params:
            self.mean_time_E = params['mean_time_E']
        else:
            self.mean_time_E = params['mean_time_exposed']

        if 'mean_time_pre_ID' in params:
            self.mean_time_pre_ID = params['mean_time_pre_ID']
        else:
            self.mean_time_pre_ID = 0

        # update so reference in params doesn't include a lambda -sw
        self.sample_QI_exit_count_p = params['sample_QI_exit_function_param']
        self.sample_QS_exit_count_p = params['sample_QS_exit_function_param']

        # parameters governing distribution over transition out of
        # each infection state
        self.exposed_infection_p = params['exposed_infection_p']
        self.daily_contacts_lambda = params['expected_contacts_per_day']

        # probability that a susceptible individual gets infected from the 'outside' on any given day
        self.daily_outside_infection_p = params['daily_outside_infection_p']

        # mild_severity_levels is the number of severity levels that are contained within the mild class.
        # We assume that they are the first entries in teh severity_prevalence array
        # severity_prevalence is an array that has the distribution of severity levels for any infected patient
        self.mild_severity_levels = params['mild_severity_levels']
        self.severity_prevalence = params['severity_prevalence']
        self.mild_symptoms_p = np.sum(self.severity_prevalence[:self.mild_severity_levels])

        # parameters governing symptomatic daily self reporting
        self.mild_self_report_p = params['mild_symptoms_daily_self_report_p']
        self.severe_self_report_p = params['severe_symptoms_daily_self_report_p']

        # parameters governing test protocol
        use_asymptomatic_testing = params['use_asymptomatic_testing']
        if use_asymptomatic_testing:
            self.days_between_tests = params['days_between_tests']
            self.test_pop_fraction = params['test_population_fraction']
            self.test_QFNR = params['test_protocol_QFNR']
            self.test_QFPR = params['test_protocol_QFPR']
            self.contact_trace_testing_frac = params['contact_trace_testing_frac']
        else:
            self.days_between_tests = 300
            self.test_pop_fraction = 0

            self.test_QFNR = 0.19
            self.test_QFPR = 0.005
            self.contact_trace_testing_frac = 1

        self.perform_contact_tracing = params['perform_contact_tracing']
        self.contact_tracing_delay = params['contact_tracing_delay']

        # new parameters governing contact tracing
        # these can be fractions -- we will round them to integers in the code
        self.cases_isolated_per_contact = params['cases_isolated_per_contact']
        self.cases_quarantined_per_contact = params['cases_quarantined_per_contact']

        # flag governing meaning of the pre-ID state
        self.pre_ID_state = params['pre_ID_state']
        assert(self.pre_ID_state in ['infectious','detectable'])

        # parameters governing initial state of simulation
        self.pop_size = params['population_size']
        self.init_E_count = params['initial_E_count']
        self.init_pre_ID_count = params['initial_pre_ID_count']
        self.init_ID_count = params['initial_ID_count']
        self.init_SyID_mild_count = params['initial_SyID_mild_count']
        self.init_SyID_severe_count = params['initial_SyID_severe_count']

        self.init_ID_prevalence = params['initial_ID_prevalence']
        if 'init_ID_prevalence_stochastic' in params:
            self.init_ID_prevalence_stochastic = params['init_ID_prevalence_stochastic']
        else:
            self.init_ID_prevalence_stochastic = False

        if 'arrival_testing_proportion' in params:
            self.arrival_testing_proportion = params['arrival_testing_proportion']
        else:
            self.arrival_testing_proportion = self.test_pop_fraction

        self.init_S_count = self.pop_size - self.init_E_count - \
            self.init_pre_ID_count - self.init_ID_count - \
            self.init_SyID_mild_count - self.init_SyID_severe_count
        assert(self.init_S_count >= 0)

        self.contact_rate_multiplier = params['contact_rate_multiplier']

        # instantiate state variables and relevant simulation variables
        self.reset_initial_state()

    def reset_initial_state(self):
        if self.init_ID_prevalence:
            if self.init_ID_prevalence_stochastic:
                init_ID_count = np.random.binomial(self.pop_size, self.init_ID_prevalence)
            else:
                init_ID_count = ceil(self.pop_size * self.init_ID_prevalence)
        else:
            init_ID_count = self.init_ID_count

        self.S = self.init_S_count + self.init_ID_count - init_ID_count

        # all of the following state vectors have the following convention:
        # state[k] is how many people have k days left to go.
        # E_sample = self.sample_E_times(self.init_E_count)
        E_sample = self.multinomial_sample(self.init_E_count, self.max_time_E, self.mean_time_E)
        self.E = E_sample[1:]

        # pre_ID_sample = self.sample_pre_ID_times(self.init_pre_ID_count + E_sample[0])
        pre_ID_sample = self.multinomial_sample(self.init_pre_ID_count + E_sample[0], self.max_time_pre_ID, self.mean_time_pre_ID)
        self.pre_ID = pre_ID_sample[1:]

        # ID_sample = self.sample_ID_times(init_ID_count + pre_ID_sample[0])
        ID_sample = self.multinomial_sample(init_ID_count + pre_ID_sample[0], self.max_time_ID, self.mean_time_ID)
        self.ID = ID_sample[1:]

        additional_mild = np.random.binomial(ID_sample[0], self.mild_symptoms_p)
        additional_severe = ID_sample[0] - additional_mild

        # SyID_mild_sample = self.sample_SyID_mild_times(self.init_SyID_mild_count + additional_mild)
        SyID_mild_sample = self.multinomial_sample(self.init_SyID_mild_count + additional_mild, self.max_time_SyID_mild, self.mean_time_SyID_mild)
        self.SyID_mild = SyID_mild_sample[1:]

        # SyID_severe_sample = self.sample_SyID_severe_times(self.init_SyID_severe_count + additional_severe)
        SyID_severe_sample = self.multinomial_sample(self.init_SyID_severe_count + additional_severe, self.max_time_SyID_severe, self.mean_time_SyID_severe)
        self.SyID_severe = SyID_severe_sample[1:]

        # contact_trace_queue[k] are the number of quarantined individuals who have k
        # days remaining until the results from their contact trace comes in
        self.contact_trace_queue = [0] * (self.contact_tracing_delay + 1)

        self.QS = 0
        self.QI = 0

        self.QI_mild = 0
        self.QI_severe = 0

        self.R = SyID_mild_sample[0] + SyID_severe_sample[0]
        self.R_mild = SyID_mild_sample[0]
        self.R_severe = SyID_severe_sample[0]

        self.cumulative_outside_infections = 0
        var_labels = self.get_state_vector_labels()
        self.sim_df = pd.DataFrame(columns=var_labels)
        self._append_sim_df()
        self.current_day = 0
        self.last_test_day = -1
        self.new_QI_from_last_test = 0
        self.new_QS_from_last_test = 0
        self.new_QI_from_self_reports = 0

    def run_new_trajectory(self, T):
        self.reset_initial_state()
        for _ in range(T):
            self.step()

        for i in range(len(self.severity_prevalence)):
            if i < self.mild_severity_levels:
                # Mild severity
                self.sim_df['severity_'+str(i)] = self.sim_df['cumulative_mild'] * (self.severity_prevalence[i] / self.mild_symptoms_p)
            else:
                # Severe symptoms
                self.sim_df['severity_'+str(i)] = self.sim_df['cumulative_severe'] * (self.severity_prevalence[i] / (1 - self.mild_symptoms_p))

        return self.sim_df, self.params

    def step_contact_trace(self, new_QI):
        """ resolve contact traces at the front of the queue and add new QIs to the back
        of the contact trace queue"""

        # update the contact trace queue
        self.contact_trace_queue[self.contact_tracing_delay] += new_QI
        resolve_today_QI = self.contact_trace_queue[0]
        self._shift_contact_queue()

        # compute how many cases we find
        # total_contacts = int(resolve_today_QI * self.contact_trace_infectious_window \
        #                                * self.daily_contacts_lambda)
        # total_contacts_traced = np.random.binomial(total_contacts, self.contact_tracing_c)
        # total_cases_isolated = np.random.binomial(total_contacts_traced, self.exposed_infection_p)
        # total_contacts_quarantined = min(self.S, total_contacts_traced - total_cases_isolated)

        total_contacts_quarantined = min(self.S, int(self.cases_quarantined_per_contact * resolve_today_QI))
        # add susceptible people to the quarantine state
        self.S = self.S - total_contacts_quarantined
        self.QS = self.QS + total_contacts_quarantined

        total_cases_isolated = int(self.cases_isolated_per_contact * resolve_today_QI)

        # trace these cases across E, pre-ID and ID states

        initial_isolations = total_cases_isolated

        leave_E = int(min(sum(self.E), total_cases_isolated))
        self._trace_E_queue(leave_E)
        total_cases_isolated -= leave_E

        leave_pre_ID = min(sum(self.pre_ID), total_cases_isolated)
        self._trace_pre_ID_queue(leave_pre_ID)
        total_cases_isolated -= leave_pre_ID

        leave_ID = min(sum(self.ID), total_cases_isolated)
        self._trace_ID_queue(leave_ID)
        total_cases_isolated -= leave_ID

        leave_SyID_severe = min(sum(self.SyID_severe), total_cases_isolated)
        self._trace_SyID_severe_queue(leave_SyID_severe)
        total_cases_isolated -= leave_SyID_severe

        leave_SyID_mild = min(sum(self.SyID_mild), total_cases_isolated)
        self._trace_SyID_mild_queue(leave_SyID_mild)
        total_cases_isolated -= leave_SyID_mild

        #print("initial isolations: {}, final isolations: {}".format(initial_isolations, total_cases_isolated))

    def _trace_SyID_severe_queue(self, leave_SyID_severe):
        assert(leave_SyID_severe <= sum(self.SyID_severe))
        self.QI = self.QI + leave_SyID_severe
        self.QI_severe += leave_SyID_severe
        idx = self.max_time_SyID_severe - 1
        while leave_SyID_severe > 0:
            leave_SyID_severe_at_idx = min(self.SyID_severe[idx], leave_SyID_severe)
            self.SyID_severe[idx] -= leave_SyID_severe_at_idx
            leave_SyID_severe -= leave_SyID_severe_at_idx
            idx -= 1

    def _trace_SyID_mild_queue(self, leave_SyID_mild):
        assert(leave_SyID_mild <= sum(self.SyID_mild))
        self.QI = self.QI + leave_SyID_mild
        self.QI_mild += leave_SyID_mild
        idx = self.max_time_SyID_mild - 1
        while leave_SyID_mild > 0:
            leave_SyID_mild_at_idx = min(self.SyID_mild[idx], leave_SyID_mild)
            self.SyID_mild[idx] -= leave_SyID_mild_at_idx
            leave_SyID_mild -= leave_SyID_mild_at_idx
            idx -= 1

    def _trace_E_queue(self, leave_E):
        assert(leave_E <= sum(self.E))
        self.QI = self.QI + leave_E
        leave_E_mild = np.random.binomial(leave_E, self.mild_symptoms_p)
        leave_E_severe = leave_E - leave_E_mild
        self.QI_mild += leave_E_mild
        self.QI_severe += leave_E_severe
        idx = self.max_time_E - 1
        while leave_E > 0:
            leave_E_at_idx = min(self.E[idx], leave_E)
            self.E[idx] -= leave_E_at_idx
            leave_E -= leave_E_at_idx
            idx -= 1

    def _trace_pre_ID_queue(self, leave_pre_ID):
        assert(leave_pre_ID <= sum(self.pre_ID))
        self.QI = self.QI + leave_pre_ID
        leave_pre_ID_mild = np.random.binomial(leave_pre_ID, self.mild_symptoms_p)
        leave_pre_ID_severe = leave_pre_ID - leave_pre_ID_mild
        self.QI_mild += leave_pre_ID_mild
        self.QI_severe += leave_pre_ID_severe
        idx = self.max_time_pre_ID - 1
        while leave_pre_ID > 0:
            leave_pre_ID_at_idx = min(self.pre_ID[idx], leave_pre_ID)
            self.pre_ID[idx] -= leave_pre_ID_at_idx
            leave_pre_ID -= leave_pre_ID_at_idx
            idx -= 1

    def _trace_ID_queue(self, leave_ID):
        assert(leave_ID <= sum(self.ID))
        self.QI = self.QI + leave_ID
        leave_ID_mild = np.random.binomial(leave_ID, self.mild_symptoms_p)
        leave_ID_severe = leave_ID - leave_ID_mild
        self.QI_mild += leave_ID_mild
        self.QI_severe += leave_ID_severe
        idx = self.max_time_ID - 1
        while leave_ID > 0:
            leave_ID_at_idx = min(self.ID[idx], leave_ID)
            self.ID[idx] -= leave_ID_at_idx
            leave_ID -= leave_ID_at_idx
            idx -= 1

    def _shift_contact_queue(self):
        idx = 0
        while idx <= self.contact_tracing_delay - 1:
            self.contact_trace_queue[idx] = self.contact_trace_queue[idx+1]
            idx += 1
        self.contact_trace_queue[self.contact_tracing_delay] = 0

    def run_test(self):
        """
        Execute one step of the testing logic.
        """

        # infectious_test_pop = free_infectious * self.test_pop_fraction
        # fluid_new_QI = infectious_test_pop * (1 - self.test_QFNR)

        # the probability that a free infected individual is quarantined
        # on this round of testing.

        # If arrival testing is specified, uses that on frist day, otherwise
        # all test_pop_fractions default to self.test_pop_fraction (configured in
        # param reads -SW)
        if self.current_day == 0:
            test_pop_fraction = self.arrival_testing_proportion
        else:
            test_pop_fraction = self.test_pop_fraction

        new_QI_p = test_pop_fraction * (1 - self.test_QFNR)

        # sample the number of free infected people who end up quarantined
        new_QI_from_ID = np.random.binomial(self.ID, new_QI_p)
        new_QI_from_ID_mild = np.random.binomial(new_QI_from_ID, self.mild_symptoms_p)
        new_QI_from_ID_severe = new_QI_from_ID - new_QI_from_ID_mild
        new_QI_from_SyID_mild = np.random.binomial(self.SyID_mild, new_QI_p)
        new_QI_from_SyID_severe = np.random.binomial(self.SyID_severe, new_QI_p)

        # update counts in relevant states
        self.ID = self.ID - new_QI_from_ID
        self.SyID_mild = self.SyID_mild - new_QI_from_SyID_mild
        self.SyID_severe = self.SyID_severe - new_QI_from_SyID_severe

        new_QI = sum(new_QI_from_ID) + sum(new_QI_from_SyID_mild) + sum(new_QI_from_SyID_severe)
        new_QI_mild = sum(new_QI_from_ID_mild) + sum(new_QI_from_SyID_mild)
        new_QI_severe = sum(new_QI_from_ID_severe) + sum(new_QI_from_SyID_severe)

        # do the above for pre-ID state, if it is detectable
        if self.pre_ID_state == 'detectable':
            new_QI_from_pre_ID = np.random.binomial(self.pre_ID, new_QI_p)
            new_QI_from_pre_ID_mild = np.random.binomial(new_QI_from_pre_ID, self.mild_symptoms_p)
            new_QI_from_pre_ID_severe = new_QI_from_pre_ID - new_QI_from_pre_ID_mild
            self.pre_ID = self.pre_ID - new_QI_from_pre_ID
            new_QI += sum(new_QI_from_pre_ID)
            new_QI_mild += sum(new_QI_from_pre_ID_mild)
            new_QI_severe += sum(new_QI_from_pre_ID_severe)

        # add to QI individuals from E, and from pre-ID (if state is 'infectious'), using
        # the false-positive rate for undetectable individuals
        new_QI_undetectable_p = test_pop_fraction * self.test_QFPR

        new_QI_from_E = np.random.binomial(self.E, new_QI_undetectable_p)
        new_QI_from_E_mild = np.random.binomial(new_QI_from_E, self.mild_symptoms_p)
        new_QI_from_E_severe = new_QI_from_E - new_QI_from_E_mild
        self.E = self.E - new_QI_from_E
        new_QI += sum(new_QI_from_E)
        new_QI_mild += sum(new_QI_from_E_mild)
        new_QI_severe += sum(new_QI_from_E_severe)

        if self.pre_ID_state == 'infectious':
            new_QI_from_pre_ID = np.random.binomial(self.pre_ID, new_QI_undetectable_p)
            new_QI_from_pre_ID_mild = np.random.binomial(new_QI_from_pre_ID, self.mild_symptoms_p)
            new_QI_from_pre_ID_severe = new_QI_from_pre_ID - new_QI_from_pre_ID_mild
            self.pre_ID = self.pre_ID - new_QI_from_pre_ID
            new_QI += sum(new_QI_from_pre_ID)
            new_QI_mild += sum(new_QI_from_pre_ID_mild)
            new_QI_severe += sum(new_QI_from_pre_ID_severe)

        # add to QS individuals from S, due to false positives
        new_QS_p = test_pop_fraction * self.test_QFPR
        # sample number of free susceptible people who become quarantined
        new_QS_from_S = np.random.binomial(self.S, new_QS_p)
        self.S = self.S - new_QS_from_S

        # update QS and QI
        self.QS = self.QS + new_QS_from_S
        self.QI = self.QI + new_QI
        self.QI_mild += new_QI_mild
        self.QI_severe += new_QI_severe

        self.new_QI_from_last_test = new_QI
        self.new_QS_from_last_test = new_QS_from_S

        return new_QI

    def isolate_self_reports(self):
        mild_self_reports = np.random.binomial(self.SyID_mild, self.mild_self_report_p)
        self.SyID_mild = self.SyID_mild - mild_self_reports
        new_QI = sum(mild_self_reports)
        new_QI_mild = sum(mild_self_reports)

        severe_self_reports = np.random.binomial(self.SyID_severe, self.severe_self_report_p)
        self.SyID_severe = self.SyID_severe - severe_self_reports
        new_QI += sum(severe_self_reports)
        new_QI_severe = sum(severe_self_reports)

        self.QI = self.QI + new_QI
        self.QI_mild += new_QI_mild
        self.QI_severe += new_QI_severe

        self.new_QI_from_self_reports = new_QI
        return new_QI

    def step(self):
        """ simulate a single day in the progression of the disease """

        new_QI = 0
        new_contact_traces = 0
        # do testing logic first
        if self.current_day - self.last_test_day >= self.days_between_tests:
            self.last_test_day = self.current_day
            new_QI += self.run_test()
            new_contact_traces += int(self.contact_trace_testing_frac * new_QI)

        # resolve symptomatic self-reporting
        new_self_reports = self.isolate_self_reports()
        new_QI += new_self_reports
        new_contact_traces += new_self_reports

        # do contact tracing
        if self.perform_contact_tracing:
            self.step_contact_trace(new_contact_traces)

        # simulate number of contacts between free infectious & free susceptible:
        free_infectious = 0

        if self.pre_ID_state == 'infectious':
            free_infectious += sum(self.pre_ID)

        free_infectious += sum(self.ID) + sum(self.SyID_mild) + sum(self.SyID_severe)

        free_susceptible = self.S
        free_tot = free_infectious + free_susceptible + self.R + sum(self.E) 

        if self.pre_ID_state == 'detectable':
            free_tot += sum(self.pre_ID)

        if free_tot == 0:
            poisson_param = 0
        else:
            poisson_param = free_infectious * self.daily_contacts_lambda * free_susceptible / free_tot
        n_contacts = np.random.poisson(poisson_param)
        # n_contacts = int(free_infectious * free_susceptible / free_tot * np.random.geometric(1/self.daily_contacts_lambda))

        # sample number of new E cases from 'inside' contacts
        new_E_from_inside = min(np.random.binomial(n_contacts, self.exposed_infection_p), self.S)

        # sample number of new E cases from 'outside' infection
        new_E_from_outside = np.random.binomial(self.S - new_E_from_inside, self.daily_outside_infection_p)
        self.cumulative_outside_infections += new_E_from_outside

        new_E = new_E_from_inside + new_E_from_outside
        self.S -= new_E

        # update E queue and record new pre-ID cases
        # new_E_times = self.sample_E_times(new_E)
        new_E_times = self.multinomial_sample(new_E, self.max_time_E, self.mean_time_E)
        new_pre_ID = self.E[0] + new_E_times[0]
        self._shift_E_queue()
        self.E = self.E + new_E_times[1:]

        # sample times of new pre-ID cases / update pre-ID queue/ record new ID cases
        # new_pre_ID_times = self.sample_pre_ID_times(new_pre_ID)
        new_pre_ID_times = self.multinomial_sample(new_pre_ID, self.max_time_pre_ID, self.mean_time_pre_ID)
        new_ID = self.pre_ID[0] + new_pre_ID_times[0]
        self._shift_pre_ID_queue()
        self.pre_ID = self.pre_ID + new_pre_ID_times[1:]

        # sample times of new ID cases / update ID queue/ record new SyID cases
        # new_ID_times = self.sample_ID_times(new_ID)
        new_ID_times = self.multinomial_sample(new_ID, self.max_time_ID, self.mean_time_ID)
        new_SyID = self.ID[0] + new_ID_times[0]
        self._shift_ID_queue()
        self.ID = self.ID + new_ID_times[1:]

        # decompose new_SyID into mild and severe
        new_SyID_mild = np.random.binomial(new_SyID, self.mild_symptoms_p)
        new_SyID_severe = new_SyID - new_SyID_mild

        # samples times of new SyID mild cases/ update mild queue/ record new R cases
        # new_SyID_mild_times = self.sample_SyID_mild_times(new_SyID_mild)
        new_SyID_mild_times = self.multinomial_sample(new_SyID_mild, self.max_time_SyID_mild, self.mean_time_SyID_mild)
        new_R_from_mild = self.SyID_mild[0] + new_SyID_mild_times[0]
        self._shift_SyID_mild_queue()
        self.SyID_mild = self.SyID_mild + new_SyID_mild_times[1:]

        # same as above, but for the severe symptom queue
        # new_SyID_severe_times = self.sample_SyID_severe_times(new_SyID_severe)
        new_SyID_severe_times = self.multinomial_sample(new_SyID_severe, self.max_time_SyID_severe, self.mean_time_SyID_severe)
        new_R_from_severe = self.SyID_severe[0] + new_SyID_severe_times[0]
        self._shift_SyID_severe_queue()
        self.SyID_severe = self.SyID_severe + new_SyID_severe_times[1:]

        # sample number of people who leave quarantine-I/ resolve new R cases
        # leave_QI = self.sample_QI_exit_count(self.QI)
        leave_QI = self.binomial_exit_function(self.QI, self.sample_QI_exit_count_p)
        if leave_QI == 0:
            leave_QI_mild = 0
            leave_QI_severe = 0
        else:
            leave_QI_mild = min(np.random.binomial(leave_QI, self.QI_mild / self.QI), self.QI_mild)
            leave_QI_severe = min(leave_QI - leave_QI_mild, self.QI_severe)
            leave_QI = leave_QI_mild + leave_QI_severe
        self.QI -= leave_QI
        self.QI_mild -= leave_QI_mild
        self.QI_severe -= leave_QI_severe
        self.R += leave_QI + new_R_from_mild + new_R_from_severe
        self.R_mild += leave_QI_mild + new_R_from_mild
        self.R_severe += leave_QI_severe + new_R_from_severe
        # leave_QS = self.sample_QS_exit_count(self.QS)
        leave_QS = self.binomial_exit_function(self.QS, self.sample_QS_exit_count_p)
        self.QS -= leave_QS
        self.S += leave_QS

        self._append_sim_df()

        self.current_day += 1

    # add new_E people to the infections queue from the S queue.
    # this function is written to support the companion multi-group simulation
    def add_new_infections(self, new_E):

        new_E = min(self.S, new_E)

        self.S = self.S - new_E

        # in theory it is possible for someone to go from new_E to R in a single step,
        # so we have to pass through all the states...
        # new_E_times = self.sample_E_times(new_E)
        new_E_times = self.multinomial_sample(new_E, self.max_time_E, self.mean_time_E)
        new_pre_ID = new_E_times[0]
        self.E = self.E + new_E_times[1:]

        # sample times of new pre-ID cases / update pre-ID queue/ record new ID cases
        # new_pre_ID_times = self.sample_pre_ID_times(new_pre_ID)
        new_pre_ID_times = self.multinomial_sample(new_pre_ID, self.max_time_pre_ID, self.mean_time_pre_ID)
        new_ID = new_pre_ID_times[0]
        self.pre_ID = self.pre_ID + new_pre_ID_times[1:]

        # sample times of new ID cases / update ID queue/ record new SyID cases
        # new_ID_times = self.sample_ID_times(new_ID)
        new_ID_times = self.multinomial_sample(new_ID, self.max_time_ID, self.mean_time_ID)
        new_SyID = new_ID_times[0]
        self.ID = self.ID + new_ID_times[1:]

        # decompose new_SyID into mild and severe
        new_SyID_mild = np.random.binomial(new_SyID, self.mild_symptoms_p)
        new_SyID_severe = new_SyID - new_SyID_mild

        # samples times of new SyID mild cases/ update mild queue/ record new R cases
        # new_SyID_mild_times = self.sample_SyID_mild_times(new_SyID_mild)
        new_SyID_mild_times = self.multinomial_sample(new_SyID_mild, self.max_time_SyID_mild, self.mean_time_SyID_mild)
        new_R_from_mild = new_SyID_mild_times[0]
        self.SyID_mild = self.SyID_mild + new_SyID_mild_times[1:]

        # same as above, but for the severe symptom queue
        # new_SyID_severe_times = self.sample_SyID_severe_times(new_SyID_severe)
        new_SyID_severe_times = self.multinomial_sample(new_SyID_severe, self.max_time_SyID_severe, self.mean_time_SyID_severe)
        new_R_from_severe = new_SyID_severe_times[0]
        self.SyID_severe = self.SyID_severe + new_SyID_severe_times[1:]

        self.R += new_R_from_mild + new_R_from_severe
        self.R_mild += new_R_from_mild
        self.R_severe += new_R_from_severe

    def _append_sim_df(self):
        self.generate_cumulative_stats()
        data = self.get_current_state_vector()
        labels = self.get_state_vector_labels()
        new_row_df = pd.DataFrame([data], columns=labels)
        self.sim_df = self.sim_df.append(new_row_df, ignore_index=True)
        # print(sum(data), sum(data[-1*(len(self.severity_prevalence)+2):]))
        # print(labels[-1*(len(self.severity_prevalence)+2):])
        assert(self.QI_mild + self.QI_severe == self.QI)
        assert(self.R_mild + self.R_severe == self.R)
        assert(min(self.QI_mild, self.QI_severe, self.R_mild, self.R_severe) >= 0)
        if abs(sum(data) - sum(data[-3:]) - self.pop_size) > 0.0001:
            raise(Exception("population has shrunk"))
        if np.sum(data < 0) > 0:
            raise(Exception("negative category size"))

    def _shift_E_queue(self):
        idx = 0
        while idx <= self.max_time_E - 2:
            self.E[idx] = self.E[idx+1]
            idx += 1
        self.E[self.max_time_E - 1] = 0

    def _shift_pre_ID_queue(self):
        idx = 0
        while idx <= self.max_time_pre_ID - 2:
            self.pre_ID[idx] = self.pre_ID[idx+1]
            idx += 1
        self.pre_ID[self.max_time_pre_ID - 1] = 0

    def _shift_ID_queue(self):
        idx = 0
        while idx <= self.max_time_ID - 2:
            self.ID[idx] = self.ID[idx+1]
            idx += 1
        self.ID[self.max_time_ID - 1] = 0

    def _shift_SyID_mild_queue(self):
        idx = 0
        while idx <= self.max_time_SyID_mild - 2:
            self.SyID_mild[idx] = self.SyID_mild[idx+1]
            idx += 1
        self.SyID_mild[self.max_time_SyID_mild - 1] = 0

    def _shift_SyID_severe_queue(self):
        idx = 0
        while idx <= self.max_time_SyID_severe - 2:
            self.SyID_severe[idx] = self.SyID_severe[idx+1]
            idx += 1
        self.SyID_severe[self.max_time_SyID_severe - 1] = 0

    def get_current_state_vector(self):
        return np.concatenate([
            [self.S], [self.QS], [self.QI], [self.R],
            self.E, self.pre_ID, self.ID, self.SyID_mild, self.SyID_severe,
            [self.cumulative_mild], [self.cumulative_severe], [self.cumulative_outside_infections]
            ])

    def get_state_vector_labels(self):
        return ['S', 'QS', 'QI', 'R'] + \
                ['E_{}'.format(x) for x in range(self.max_time_E)] + \
                ['pre_ID_{}'.format(x) for x in range(self.max_time_pre_ID)] + \
                ['ID_{}'.format(x) for x in range(self.max_time_ID)] + \
                ['SyID_mild_{}'.format(x) for x in range(self.max_time_SyID_mild)] + \
                ['SyID_severe_{}'.format(x) for x in range(self.max_time_SyID_severe)] + \
                ['cumulative_mild', 'cumulative_severe', 'cumulative_outside_infections']

    def generate_cumulative_stats(self):
        self.cumulative_mild = self.QI_mild + sum(self.SyID_mild) + self.R_mild
        self.cumulative_severe = self.QI_severe + sum(self.SyID_severe) + self.R_severe

        # self.severity = list()
        # for i in range(self.mild_severity_levels):
        #     self.severity.append((self.severity_prevalence[i] / self.mild_symptoms_p ) * self.cumulative_mild)
        # for i in range(len(self.severity_prevalence) - self.mild_severity_levels):
        #     self.severity.append((self.severity_prevalence[i + self.mild_severity_levels]) / (1 - self.mild_symptoms_p) * self.cumulative_severe)

    def update_severity_levels(self):
        for i in range(len(self.severity_prevalence)):
            if i < self.mild_severity_levels:
                self.sim_df['severity_'+str(i)] = self.sim_df['cumulative_mild'] * (self.severity_prevalence[i] / self.mild_symptoms_p)
            else:
                self.sim_df['severity_'+str(i)] = self.sim_df['cumulative_severe'] * (self.severity_prevalence[i] / (1 - self.mild_symptoms_p))

#################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run multiple simulations using multiprocessing')

    parser.add_argument('multigroup_file',
                        help='yaml file specifying the parameters for the multigroup simulation')

    parser.add_argument('-o', '--outputdir', default=BASE_DIRECTORY,
                        help='directory to store simulation output')

    parser.add_argument('-V', '--verbose', action='store_true', help='include verbose output')

    parser.add_argument('-s', '--scenarios', nargs='+', required=False,
                        help='list of YAML config files specifying base sets of scenario parameters to use')

    parser.add_argument('-p', '--param-to-vary', action='append',
                        help='which param(s) should be varied in the corresponding sensitivity sims', required=False)

    parser.add_argument('-v', '--values', required=False, nargs='+', action='append',
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
