""" Mutual Information Gap (MIG). """

from ..utils import *


def compute_mig(ground_truth_data, representation_function, random_state, num_train=3000, batch_size=16):
    """ Computes the Mutual Information Gap (MIG).
    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
        representation for each observation: (batch, x_shapes) -> (nz, batch)
    :param random_state: Numpy random state used for randomness.
    :param num_train: Number of points used for training.
    :param batch_size: Batch size for sampling.
    :return: Dict with average MIG: { discrete_mig: mig_score (scalar) }.
    """
    # Get mu:(nz, num_train), labels:(num_factors, num_train)
    mus_train, ys_train = generate_batch_factor_code(ground_truth_data, representation_function, num_train, random_state, batch_size)
    assert mus_train.shape[1] == num_train
    return _compute_mig(mus_train, ys_train)


def _compute_mig(mus_train, ys_train, num_bins=20, discretizer_fn=histogram_discretize):
    """ Computes score based on both training and testing codes and factors.
    :param mus_train: (nz, num_train)
    :param ys_train: (num_factors, num_train)
    :param num_bins:
    :param discretizer_fn:
    :return: { discrete_mig: mig_score (scalar), discrete_migsup: migsup_score (scalar) }.
    """
    score_dict = {}
    # Get discretized mu. (nz, num_train)
    discretized_mus = discretizer_fn(mus_train, num_bins=num_bins)
    # 1. Get discretized I(Z_j;V_k). (nz, num_factors)
    m = discrete_mutual_info(discretized_mus, ys_train)
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    # 2. Get discretized H(V_k). (num_factors, )
    entropy = discrete_entropy(ys_train)
    # Calculate
    # (1) MIG.
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["mig"] = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    # (2) MIGSUP.
    sorted_m = np.sort(m / (entropy[np.newaxis]+1e-8), axis=1)[:, ::-1]
    score_dict["migsup"] = np.mean(sorted_m[:, 0] - sorted_m[:, 1])
    # Return
    return score_dict
