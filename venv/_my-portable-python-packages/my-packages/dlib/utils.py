""" Utility functions that are useful for the different metrics. """

import sklearn
import numpy as np


def generate_batch_factor_code(ground_truth_data, representation_function, num_points, random_state, batch_size):
    """ Sample a single training sample based on a mini-batch of ground-truth data.
    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observation as input and outputs a representation.
    :param num_points: Number of points to sample.
    :param random_state: Numpy random state used for randomness.
    :param batch_size: Batchsize to sample points.
    :return:
        representations: Codes (nz, num_points)-np array.
        factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations, factors, i = None, None, 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = ground_truth_data.sample(num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations, representation_function(current_observations)))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)


def obtain_representation(observations, representation_function, batch_size):
    """ Obtain representations from observations.
    :param observations: Observations for which we compute the representation.
    :param representation_function: Function that takes observation as input and outputs a representation.
    :param batch_size: Batch size to compute the representation.
    :return: representations: Codes (num_codes, num_points)-Numpy array.
    """
    representations = None
    num_points = observations.shape[0]
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_observations = observations[i:i + num_points_iter]
        if i == 0: representations = representation_function(current_observations)
        else: representations = np.vstack((representations, representation_function(current_observations)))
        i += num_points_iter
    return np.transpose(representations)


# ----------------------------------------------------------------------------------------------------------------------
# Discretize
# ----------------------------------------------------------------------------------------------------------------------

def histogram_discretize(target, num_bins):
    """ Discretization based on histograms (along the dim1).
    :param target: (dim0, dim1)
    :param num_bins:
    :return: (dim0, dim1)
    """
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(mus, ys):
    """ Compute discrete mutual information.
    :param mus: (nz, num_train).
    :param ys: (num_factors, num_train).
    :return: (nz, num_factors).
    """
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """ Compute discrete entropy.
    :param ys: (num_factors, num_train)
    :return: (num_factors, )
    """
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h
