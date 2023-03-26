
import numpy as np


class DatasetSampling(object):
    """
    Disentanglement dataset for sampling. Wrapped from disentangling dataset.
    """
    def __init__(self, disentangling_dataset, specified_factor_indices=None):
        """
        :param disentangling_dataset: Protocols:
            - [property] factors: { name1: values1(list), name2: values2(list), ... } (OrderedDict).
            - [property] n_factors_values: [num_factor1_values, num_factor2_values, ...].
            - [method] get_observations_from_factors(factors): get observations (x) from given factors.
                * factors: np.array. (n_factors, ) or (batch, n_factors).
                * return (x_shape, ) or (batch, x_shape).
        :param specified_factor_indices: List of (n_specified_factors, )
        """
        self._dataset = disentangling_dataset
        # Configs.
        self._specified_factor_indices = specified_factor_indices if specified_factor_indices is not None else \
            list(range(len(self._dataset.factors)))
        self._observation_factor_indices = [i for i in range(len(self._dataset.factors)) if i not in self._specified_factor_indices]
        
    @property
    def num_factors(self):
        return len(self._specified_factor_indices)

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self._dataset.n_factors_values[i], size=num)

    def sample_factors(self, num, random_state):
        """ Sample a batch of factors Y.
        :return (num, n_specified_factors).
        """
        factors = np.zeros(shape=(num, len(self._specified_factor_indices)), dtype=np.int64)
        for pos, i in enumerate(self._specified_factor_indices):
            factors[:, pos] = self._sample_factor(i, num, random_state)
        return factors

    def sample_observations_from_factors(self, factors, random_state):
        """ Sample a batch of observations X given a batch of factors Y.
        :param factors: (n_samples, n_specified_factors)
        :param random_state:
        :return
        """
        # --------------------------------------------------------------------------------------------------------------
        # Get factors. (n_samples, n_factors)
        # --------------------------------------------------------------------------------------------------------------
        n_samples = len(factors)
        # 1. Init results.
        all_factors = np.zeros(shape=(n_samples, len(self._dataset.factors)), dtype=np.int64)
        # 2. Set given factors.
        all_factors[:, self._specified_factor_indices] = factors
        # 3. Complete all the other factors.
        for i in self._observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, n_samples, random_state)
        # --------------------------------------------------------------------------------------------------------------
        # Get observations.
        # --------------------------------------------------------------------------------------------------------------
        return self._dataset.get_observations_from_factors(all_factors)

    def sample(self, num, random_state):
        """ Sample a batch of factors Y and observations X. """
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def sample_observations(self, num, random_state):
        """ Sample a batch of observations X. """
        return self.sample(num, random_state)[1]
