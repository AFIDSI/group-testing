import numpy as np
from analysis_helpers import poisson_waiting_function
from subdivide_severity import subdivide_severity

prob_severity_given_age_v1 = np.array([[0.1, 0.89, 0.01, 0],\
                                    [0.07, 0.80, 0.10, 0.03],\
                                    [0.07, 0.76, 0.10, 0.07],\
                                    [0.07, 0.70, 0.13, 0.10],\
                                    [0.05, 0.55, 0.2, 0.2]])

prob_severity_given_age_v2 = np.array([[0.35, 0.63895, 0.00863, 0.00242],\
                                    [0.35, 0.63895, 0.00863, 0.00242],\
                                    [0.35, 0.62075, 0.020709, 0.008541],\
                                    [0.35, 0.6019, 0.03521, 0.01289],\
                                    [0.35, 0.6019, 0.03521, 0.01289]])

prob_severity_given_age_v3 = np.array([[0.17, 0.816, 0.011, 0.003],\
                                    [0.52, 0.472, 0.006, 0.002],\
                                    [0.31, 0.659, 0.022, 0.009],\
                                    [0.13, 0.806, 0.047, 0.017],\
                                    [0.13, 0.806, 0.047, 0.017]])

prob_infection = np.array([0.018, 0.022, 0.029, 0.042, 0.042])
prob_age = np.array([0, 0.85808522, 0.13170574, 0.00878566, 0.00142338])

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


# put the config-dict defining code inside a static methods
# so that it is not immediately run at import-time
class ParamConfig:
    @staticmethod
    def load_config(time_period, use_testing, assn_type):
        assert(time_period in ['june', 'fall'])
        assert(type(use_testing) == type(True))
        assert(assn_type in ['optimistic', 'nominal', 'pessimistic'])
        orig_severity_prevalence = subdivide_severity(prob_severity_given_age_v3, 
                                                    prob_infection, prob_age)


        if assn_type == 'optimistic':
            assn_num = 0
        elif assn_type == 'nominal':
            assn_num = 1
        elif assn_type == 'pessimistic':
            assn_num = 2

        asymptomatic_pct_mult = (0.2/0.35, 1, 0.5/0.35)[assn_num]
        new_asymptomatic_pct = asymptomatic_pct_mult * orig_severity_prevalence[0]
        severity_prevalence = update_sev_prevalence(orig_severity_prevalence, new_asymptomatic_pct)


        mean_time_ID = (2.5, 3, 3.5)[assn_num]
        max_time_ID = int(5 + mean_time_ID)

        mean_time_Sy = (10, 12, 14)[assn_num]
        max_time_Sy = int(mean_time_Sy + 8)

        daily_self_report_severe = 0.18
        daily_self_report_mild = 0

        contact_trace_testing_frac = 0.5

        prevalence = (0.001, 0.0025, 0.005)[assn_num]

        if time_period == 'fall':
            daily_contacts = 8.3
            popsize = 34310
            num_isolations = (0.87, 0.92, 0.98)[assn_num]

        elif time_period == 'june':
            daily_contacts = 4.15
            popsize = 2500
            num_isolations = (0.43, 0.46, 0.49)[assn_num]


        contacts_per_trace = 7
        num_quarantines = contacts_per_trace - num_isolations
        assert(num_quarantines >= 0)
        contact_delay = (1,1,2)[assn_num]

        transmissions_per_contact = 0.026

        # asymptomatic testing parameters 

        test_qfnr = 0.19
        test_qfpr = 0.005
        if use_testing:
            test_frequency = 1
            test_daily_fraction = 0.2
        else:
            test_frequency = 300
            test_daily_fraction = 0
        
        return {
            'max_time_exposed': 4,
            'exposed_time_function': poisson_waiting_function(max_time=4, mean_time=2),
            
            'max_time_pre_ID': 4,
            'pre_ID_time_function': poisson_waiting_function(max_time=4, mean_time=0),
            
            'max_time_ID': max_time_ID,
            'ID_time_function': poisson_waiting_function(max_time=max_time_ID, mean_time=mean_time_ID),
            
            'max_time_SyID_mild': max_time_Sy,
            'SyID_mild_time_function': poisson_waiting_function(max_time_Sy, mean_time_Sy),
            
            'max_time_SyID_severe': max_time_Sy,
            'SyID_severe_time_function': poisson_waiting_function(max_time_Sy, mean_time_Sy),
            
            'sample_QI_exit_function': (lambda n: np.random.binomial(n, 0.05)),
            'sample_QS_exit_function': (lambda n: np.random.binomial(n, 0.3)),
            
            'exposed_infection_p': transmissions_per_contact,
            'expected_contacts_per_day': daily_contacts,
            
            'mild_severity_levels': 1, # hardcoded value -- 'SyMild' corresponds to severity level 1
            'severity_prevalence': severity_prevalence,
            'mild_symptoms_daily_self_report_p': daily_self_report_mild,
            'severe_symptoms_daily_self_report_p': daily_self_report_severe,
            
            'days_between_tests': test_frequency,
            'test_population_fraction': test_daily_fraction,
            
            'test_protocol_QFNR': test_qfnr,
            'test_protocol_QFPR': test_qfpr,
            
            'perform_contact_tracing': True,
            'contact_tracing_delay': contact_delay,
            'cases_isolated_per_contact': num_isolations,
            'cases_quarantined_per_contact': num_quarantines,
            'contact_trace_testing_frac': contact_trace_testing_frac,
            
            'pre_ID_state': 'detectable',
            
            'population_size': popsize,
            'initial_E_count': 0,
            'initial_pre_ID_count': 0,
            'initial_ID_count': 0,
            'initial_ID_prevalence': prevalence,
            'initial_SyID_mild_count': 0,
            'initial_SyID_severe_count': 0
        }

