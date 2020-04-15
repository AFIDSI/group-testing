import numpy as np

class population:

    """ This class describes a population of people and their infection status, optionally including information about their
    household organization and the ability to simulate infection status in an correlated way across households.
    It has methods for simulating infection status forward in time """

    def __init__(self, n, prevalence):
        # Create a population with n individuals in which the household sizes are all 1 with the given initial disease
        # prevalence

        self.infectious = np.random.rand(n) < prevalence[0]  # True if infectious
        self.quarantined = [False]*n    # True if quarantined

        # True if susceptible to the disease, i.e., has not died or developed immunity, and not currently infected
        self.susceptible = ~self.infectious

    def __init__(self, n_households, household_size_dist, prevalence, SAR, R0, d0):
        # Initialize a population with non-trivial households
        # n_households:         the number of households in the population
        # household_size_dist:  a numpy array that should sum to 1, where household_size_dist[i] gives the fraction of
        #                       households with size i+1
        # prevalence:           prevalence of the disease in the population, used to simulate initial infection status
        # SAR:                  secondary attack rate, used for simulating initial infection status and for simulating
        #                       forward in time; SAR is defined as the probability that an infection occurs among
        #                       susceptible people within a household related to a confirmed case
        #
        assert np.isclose(np.sum(household_size_dist), 1.)

        self.infectious = np.zeros(n_households)
        self.quarantined = [False] * n_households
        self.prevalence = prevalence
        self.n_households = n_households
        self.R0 = R0
        self.d0 = d0

        self.total_pop = 0

        for i in range(n_households):
            # generate household size h from household_size_dist
            h = int(np.random.choice(np.arange(1, len(household_size_dist)+1), 1, p=household_size_dist))
            self.total_pop += h
            # compute primary case probability = p*h/(1+SAR*(h-1))
            prob_prim = prevalence*h/(1+SAR*(h-1))
            self.infectious[i] = np.random.rand(1) < prob_prim
            if h > 1:
                # if there are >1 members in the household, and there is a primary case,
                # generate secondary cases from Bin(h-1, SAR); otherwise, set everyone to be uninfected
                self.infectious[i].extend(np.random.binomial(1, SAR, h-1) * self.infectious[i]==1)
                self.quarantined[i].extend([False]*(h-1))

        self.susceptible = ~self.infectious

    def __check_indices(self,x):
        # Make sure that a passed set of individual array indices are valid for our population
        # TODO: This should be adapted for households?
        assert(max(x) < self.n)
        assert(min(x) >= 0)

    def infected(self, x):
        # Given a set of indices x, return the infection status of those individuals
        # Used by tests based on this class
        self.__check_indices(x)
        return self.infectious[x]

    def get_prevalence(self):
        # Get current prevalence = (number of infectious unquarantined) / (number of unquarantined)
        infected_counts = 0
        unquarantined_counts = 0

        for i in range(self.n_households):
            infected_counts += np.sum(self.infectious[i] and ~self.quarantined[i])
            unquarantined_counts += np.sum(~self.quarantine[i])
        return infected_counts / unquarantined_counts

    def step(self):
        # Simulate one step forward in time
        # Simulate how infectious individuals infect each other
        # Unquarantined susceptible people become infected w/ probability = alpha*current prevalence
        for i in range(self.n_households):
            for j in range(len(self.quarantined[i])):
                if ~self.quarantined[i][j] and self.susceptible[i][j]:
                    self.infectious[i][j] = np.random.rand(1) < self.R0**(1/self.d0) * self.get_prevalence


    def quarantine(self, x):
        # Put into quarantine all individuals whose indices are in the list x
        self.__check_indices(x)
        self.quarantined[x] = True

    def unquarantine(self, x):
        # Remove from quarantine all individuals whose indices are in the list x
        self.__check_indices(x)
        self.quarantined[x] = False

    def num_infected(self):
        infected_counts = 0
        for i in range(self.n_households):
            infected_counts += np.sum(self.infectious[i] and ~self.quarantined[i])
        return infected_counts

    #def num_recovered_dead(self):
    # for now we haven't taken into consideration the duration of infection and recovery yet
