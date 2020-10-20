import os

import copy
import numpy as np
import pdb
import yaml

from subdivide_severity import subdivide_severity
from load_params import load_age_sev_params

# upper bound on how far the recursion can go in the yaml-depency tree
MAX_DEPTH = 5

# simulation parameters which can be included as yaml-keys but are not required
# they are set to a default value of 0 if not included in the yaml-config
DEFAULT_ZERO_PARAMS = ['initial_E_count',
                       'initial_pre_ID_count',
                       'initial_ID_count',
                       'initial_SyID_mild_count',
                       'initial_SyID_severe_count']

# yaml-keys which share the same key as the simulation parameter, and
# can be copied over one-to-one
COPY_DIRECTLY_YAML_KEYS = ['exposed_infection_p', 'expected_contacts_per_day',
                           'perform_contact_tracing', 'contact_tracing_delay',
                           'cases_isolated_per_contact', 'cases_quarantined_per_contact',
                           'use_asymptomatic_testing', 'contact_trace_testing_frac', 'days_between_tests',
                           'test_population_fraction', 'test_protocol_QFNR',
                           'test_protocol_QFPR', 'initial_ID_prevalence',
                           'population_size', 'severity_prevalence', 'age_distribution',
                           'daily_outside_infection_p', 'arrival_testing_proportion',
                           'contact_rate_multiplier'] + \
            DEFAULT_ZERO_PARAMS


def load_parameters_from_yaml(param_file):
    """
    Parses the yaml file and loads all of the elements necessary for
    initializing the simulation. Note that all groups are initialized with the
    default parameters, and overwritten with any group-level parameters that
    are specified.
    """

    with open(param_file) as file:
        simulation_parameters = yaml.safe_load(file)

        # create empty dictionary for collecting processed parameters from yaml
        base_params = {}

        base_params['ntrajectories'] = simulation_parameters['ntrajectories']

        base_params['time_horizon'] = simulation_parameters['time_horizon']

        # load interaction matrix
        base_params['interaction_matrix'] = np.array(simulation_parameters['interaction_matrix'])

        # load group names
        base_params['group_names'] = simulation_parameters['group_names']

        # load severities
        assert ('severity_prevalence' in simulation_parameters), 'Severity information not contained in {}'.format(param_file)
        severity_prevalence = prep_severity(simulation_parameters['severity_prevalence'])

        # load default parameters
        assert ('scenario_defaults' in simulation_parameters), 'scenario_defaults not contained in {}'.format(param_file)
        default_params = load_defaults(simulation_parameters['scenario_defaults'], severity_prevalence)


        # load group-specific parameterizations
        assert ('groups' in simulation_parameters), 'Group information not contained in {}'.format(param_file)

        group_definitions = {}
        for group_name, group_params in simulation_parameters['groups'].items():
            group_definitions[group_name] = load_multi_group_params(group_params, default_params)
        base_params['groups'] = group_definitions

        return base_params


####################
# HELPER FUNCTIONS #
####################


def identify_dynamic_params(base_params):
    default_param_dimensions = {'E_time_params': 2,
                                'ID_time_params': 2,
                                'Sy_time_params': 2,
                                'initial_E_count': 1,
                                'initial_pre_ID_count': 1,
                                'initial_ID_count': 1,
                                'initial_SyID_mild_count': 1,
                                'initial_SyID_severe_count': 1,
                                'exposed_infection_p': 1,
                                'expected_contacts_per_day': 1,
                                'perform_contact_tracing': 1,
                                'contact_tracing_delay': 1,
                                'cases_isolated_per_contact': 1,
                                'cases_quarantined_per_contact': 1,
                                'use_asymptomatic_testing': 1,
                                'contact_trace_testing_frac': 1,
                                'days_between_tests': 1,
                                'test_population_fraction': 1,
                                'test_protocol_QFNR': 1,
                                'test_protocol_QFPR': 1,
                                'initial_ID_prevalence': 1,
                                'population_size': 1,
                                'severity_prevalence': 1,
                                'age_distribution': 1,
                                'daily_outside_infection_p': 1,
                                'arrival_testing_proportion': 1,
                                'contact_rate_multiplier': 1}

    # grab example group
    group = base_params['groups'][list(base_params['groups'].keys())[0]]

    # check the number of lists in the age_distribution
    number_of_age_groups = np.array(group['age_distribution']).shape[-1]
    default_param_dimensions['age_distribution'] = number_of_age_groups
    default_param_dimensions['severity_prevalence'] = number_of_age_groups
    default_param_dimensions['prob_infection_by_age'] = number_of_age_groups

    # create a dictionary of dynamic params by checking the observed length against the expected
    dynamic_params = {}
    for group_name, group_params in base_params['groups'].items():
        for param_name, param_value in group_params.items():
            if type(param_value) is list:
                if len(param_value) != default_param_dimensions[param_name]:
                    # if group_name not in dynamic_params.keys():
                    #     dynamic_params[group_name] = {}
                    dynamic_params[(group_name, param_name)] = param_value

    return dynamic_params


def load_multi_group_params(group_params, default_params):
    returnable_param_set = copy.deepcopy(default_params)
    for param_name, param_value in group_params.items():
        returnable_param_set[param_name] = param_value
    return returnable_param_set


def load_defaults(default_input, severity_prevalence):
    default_params = {}

    default_params['severity_prevalence'] = severity_prevalence

    for yaml_key, val in default_input.items():

        # skip the meta-params in this for loop
        if yaml_key[0] == '_':
            continue

        if yaml_key == 'ID_time_params':
            assert(len(val) == 2)

            mean_time_ID = val[0]
            max_time_ID = val[1]

            default_params['mean_time_ID'] = mean_time_ID
            default_params['max_time_ID'] = max_time_ID

        elif yaml_key == 'E_time_params':
            assert(len(val) == 2)
            default_params['max_time_exposed'] = val[1]
            default_params['mean_time_exposed'] = val[0]

        elif yaml_key == 'Sy_time_params':
            assert(len(val) == 2)
            default_params['max_time_SyID_mild'] = val[1]
            default_params['mean_time_SyID_mild'] = val[0]
            # default_params['max_time_SyID_mild'] = val[1]

            default_params['max_time_SyID_severe'] = val[1]
            default_params['mean_time_SyID_severe'] = val[0]
            # default_params['max_time_SyID_severe'] = val[1]

        elif yaml_key == 'asymptomatic_daily_self_report_p':
            default_params['mild_symptoms_daily_self_report_p'] = val

        elif yaml_key == 'symptomatic_daily_self_report_p':
            default_params['severe_symptoms_daily_self_report_p'] = val

        elif yaml_key == 'daily_leave_QI_p':
            # default_params['sample_QI_exit_function'] = binomial_exit_function(val)  # (lambda n: np.random.binomial(n, val))
            default_params['sample_QI_exit_function_param'] = val
            # change to just pass value and reference binomial_exit_function later -sw

        elif yaml_key == 'daily_leave_QS_p':
            # default_params['sample_QS_exit_function'] = binomial_exit_function(val)  # (lambda n: np.random.binomial(n, val))
            default_params['sample_QS_exit_function_param'] = val
            # change to just pass value and reference binomial_exit_function later -sw

        elif yaml_key == 'asymptomatic_pct_mult':
            if 'severity_prevalence' not in default_params:
                raise(Exception("encountered asymptomatic_pct_mult with no corresponding severity_dist to modify"))
            new_asymptomatic_p = val * default_params['severity_prevalence'][0]
            default_params['severity_prevalence'] = update_sev_prevalence(default_params['severity_prevalence'],
                                                                          new_asymptomatic_p)

        elif yaml_key in COPY_DIRECTLY_YAML_KEYS:
            default_params[yaml_key] = val

        elif yaml_key == 'group_name':
            default_params['group_name'] = default_input['group_name']

        else:
            raise(Exception("encountered unknown parameter {}".format(yaml_key)))

    # the pre-ID state is not being used atm so fill it in with some default params here
    if 'max_time_pre_ID' not in default_params:
        default_params['max_time_pre_ID'] = 4

    # the following 'initial_count' variables are all defaulted to 0
    for paramname in DEFAULT_ZERO_PARAMS:
        if paramname not in default_params:
            default_params[paramname] = 0

    if 'pre_ID_state' not in default_params:
        default_params['pre_ID_state'] = 'detectable'

    if 'mild_severity_levels' not in default_params:
        default_params['mild_severity_levels'] = 1

    if 'contact_rate_multiplier' not in default_params:
        default_params['contact_rate_multiplier'] = 1

    return default_params


def load_params(param_file, param_file_stack=[], additional_params={}):
    # process the main params loaded from yaml, as well as the additional_params
    # optionally passed as an argument, and store them in base_params

    MAX_DEPTH = 2
    with open(param_file) as file:
        params_object = yaml.safe_load(file)

    cwd = os.getcwd()

    nwd = os.path.dirname(os.path.realpath(param_file))
    os.chdir(nwd)

    return_object = {}
    params = {}
    dynamic_params = {}

    if '_inherit_config' in params_object.keys():
        if len(param_file_stack) >= MAX_DEPTH:
            raise(Exception("yaml config dependency depth exceeded max depth"))
        new_param_file = params_object['_inherit_config']
        params = load_params(new_param_file, param_file_stack + [param_file])['params']

    if '_age_severity_config' in params_object.keys():
        age_sev_file = params_object['_age_severity_config']
        severity_dist = load_age_sev_params(age_sev_file)
        params['severity_prevalence'] = severity_dist
    else:
        severity_dist = None
    if '_scenario_name' in params_object.keys():
        params['scenario_name'] = params_object['_scenario_name']

    if param_file is not None:
        # change working-directory back
        os.chdir(cwd)

    for yaml_key, val in params_object.items():

        # skip the meta-params in this for loop
        if yaml_key[0] == '_':
            continue

        if yaml_key == 'ID_time_params':
            assert(len(val) == 2)

            mean_time_ID = val[0]
            max_time_ID = val[1]

            params['mean_time_ID'] = mean_time_ID
            params['max_time_ID'] = max_time_ID

        elif yaml_key == 'E_time_params':
            assert(len(val) == 2)
            params['max_time_exposed'] = val[1]
            params['mean_time_exposed'] = val[0]

        elif yaml_key == 'Sy_time_params':
            assert(len(val) == 2)
            params['max_time_SyID_mild'] = val[1]
            params['mean_time_SyID_mild'] = val[0]
            # params['max_time_SyID_mild'] = val[1]

            params['max_time_SyID_severe'] = val[1]
            params['mean_time_SyID_severe'] = val[0]
            # params['max_time_SyID_severe'] = val[1]

        elif yaml_key == 'asymptomatic_daily_self_report_p':
            params['mild_symptoms_daily_self_report_p'] = val

        elif yaml_key == 'symptomatic_daily_self_report_p':
            params['severe_symptoms_daily_self_report_p'] = val

        elif yaml_key == 'daily_leave_QI_p':
            # params['sample_QI_exit_function'] = binomial_exit_function(val)  # (lambda n: np.random.binomial(n, val))
            params['sample_QI_exit_function_param'] = val
            # change to just pass value and reference binomial_exit_function later -sw

        elif yaml_key == 'daily_leave_QS_p':
            # params['sample_QS_exit_function'] = binomial_exit_function(val)  # (lambda n: np.random.binomial(n, val))
            params['sample_QS_exit_function_param'] = val
            # change to just pass value and reference binomial_exit_function later -sw

        elif yaml_key == 'asymptomatic_pct_mult':
            if 'severity_prevalence' not in params:
                raise(Exception("encountered asymptomatic_pct_mult with no corresponding severity_dist to modify"))
            new_asymptomatic_p = val * params['severity_prevalence'][0]
            params['severity_prevalence'] = update_sev_prevalence(params['severity_prevalence'],
                                                                  new_asymptomatic_p)

        elif yaml_key in COPY_DIRECTLY_YAML_KEYS:
            params[yaml_key] = val

        elif yaml_key == 'parameters_to_vary':
            dynamic_params = params_object['parameters_to_vary']

        elif yaml_key == 'group_name':
            params['group_name'] = params_object['group_name']

        else:
            raise(Exception("encountered unknown parameter {}".format(yaml_key)))

    # the pre-ID state is not being used atm so fill it in with some default params here
    if 'max_time_pre_ID' not in params:
        params['max_time_pre_ID'] = 4

    # the following 'initial_count' variables are all defaulted to 0
    for paramname in DEFAULT_ZERO_PARAMS:
        if paramname not in params:
            params[paramname] = 0

    if 'pre_ID_state' not in params:
        params['pre_ID_state'] = 'detectable'

    if 'mild_severity_levels' not in params:
        params['mild_severity_levels'] = 1

    return_object['params'] = params
    return_object['dynamic_params'] = dynamic_params
    return return_object


def update_sev_prevalence(curr_prevalence_dist, new_asymptomatic_pct):
    new_dist = [new_asymptomatic_pct]
    remaining_mass = sum(curr_prevalence_dist[1:])

    # need to scale so that param_val + x * remaning_mass == 1
    scale = (1 - new_asymptomatic_pct) / remaining_mass
    idx = 1
    while idx < len(curr_prevalence_dist):
        new_dist.append(curr_prevalence_dist[idx] * scale)
        idx += 1
    assert(np.isclose(sum(new_dist), 1))
    return np.array(new_dist)


def prep_severity(age_severity_parameters):
    subparams = age_severity_parameters['prob_severity_given_age']
    prob_severity_given_age = np.array([
        subparams['agegroup1'],
        subparams['agegroup2'],
        subparams['agegroup3'],
        subparams['agegroup4'],
        subparams['agegroup5'],
    ])

    prob_infection = np.array(age_severity_parameters['prob_infection_by_age'])
    prob_age = np.array(age_severity_parameters['age_distribution'])
    return subdivide_severity(prob_severity_given_age, prob_infection, prob_age)
