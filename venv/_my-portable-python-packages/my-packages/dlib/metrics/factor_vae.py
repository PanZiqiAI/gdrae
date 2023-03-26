""" Implementation of the disentanglement metric from the FactorVAE paper. """

from ..utils import *
from six.moves import range


def compute_factor_vae(ground_truth_data, representation_function, random_state, batch_size=5, num_train=1000, num_eval=100, num_variance_estimate=100):
    """ Computes the FactorVAE disentanglement metric.
    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
        representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param batch_size: Number of points to be used to compute the training_sample.
    :param num_train: Number of points used for training.
    :param num_eval: Number of points used for evaluation.
    :param num_variance_estimate: Number of points used to estimate global variances.
    :return: Dictionary with scores:
        train_accuracy: Accuracy on training set.
        eval_accuracy: Accuracy on evaluation set.
    """
    global_variances = _compute_variances(ground_truth_data, representation_function, num_variance_estimate, random_state)
    active_dims = _prune_dims(global_variances)
    scores_dict = {}
    if not active_dims.any():
        scores_dict["train_accuracy"] = 0.
        scores_dict["eval_accuracy"] = 0.
        scores_dict["num_active_dims"] = 0
        return scores_dict
    training_votes = _generate_training_batch(ground_truth_data, representation_function, batch_size, num_train, random_state, global_variances, active_dims)
    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])
    train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
    eval_votes = _generate_training_batch(ground_truth_data, representation_function, batch_size, num_eval, random_state, global_variances, active_dims)
    eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)
    scores_dict["train_accuracy"] = train_accuracy
    scores_dict["eval_accuracy"] = eval_accuracy
    scores_dict["num_active_dims"] = len(active_dims)
    return scores_dict


def _prune_dims(variances, threshold=0.):
    """ Mask for dimensions collapsed to the prior. """
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def _compute_variances(ground_truth_data, representation_function, batch_size, random_state, eval_batch_size=64):
    """ Computes the variance for each dimension of the representation.
    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observation as input and outputs a representation.
    :param batch_size: Number of points to be used to compute the variances.
    :param random_state: Numpy random state used for randomness.
    :param eval_batch_size: Batch size used to eval representation.
    :return: Vector with the variance of each dimension.
    """
    observations = ground_truth_data.sample_observations(batch_size, random_state)
    representations = obtain_representation(observations, representation_function, eval_batch_size)
    representations = np.transpose(representations)
    assert representations.shape[0] == batch_size
    return np.var(representations, axis=0, ddof=1)


def _generate_training_sample(ground_truth_data, representation_function, batch_size, random_state, global_variances, active_dims):
    """ Sample a single training sample based on a mini-batch of ground-truth data.
    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observation as input and outputs a representation.
    :param batch_size: Number of points to be used to compute the training_sample.
    :param random_state: Numpy random state used for randomness.
    :param global_variances: Numpy vector with variances for all dimensions of representation.
    :param active_dims: Indexes of active dimensions.
    :return:
        factor_index: Index of factor coordinate to be used.
        argmin: Index of representation coordinate with the least variance.
    """
    # Select random coordinate to keep fixed.
    factor_index = random_state.randint(ground_truth_data.num_factors)
    # Sample two mini batches of latent variables.
    factors = ground_truth_data.sample_factors(batch_size, random_state)
    # Fix the selected factor across mini-batch.
    factors[:, factor_index] = factors[0, factor_index]
    # Obtain the observations.
    observations = ground_truth_data.sample_observations_from_factors(factors, random_state)
    representations = representation_function(observations)
    local_variances = np.var(representations, axis=0, ddof=1)
    argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])
    return factor_index, argmin


def _generate_training_batch(ground_truth_data, representation_function, batch_size, num_points, random_state, global_variances, active_dims):
    """ Sample a set of training samples based on a batch of ground-truth data.
    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
        representation for each observation.
    :param batch_size: Number of points to be used to compute the training_sample.
    :param num_points: Number of points to be sampled for training set.
    :param random_state: Numpy random state used for randomness.
    :param global_variances: Numpy vector with variances for all dimensions of representation.
    :param active_dims: Indexes of active dimensions.
    :return: (num_factors, dim_representation)-sized numpy array with votes.
    """
    votes = np.zeros((ground_truth_data.num_factors, global_variances.shape[0]), dtype=np.int64)
    for _ in range(num_points):
        factor_index, argmin = _generate_training_sample(ground_truth_data, representation_function, batch_size, random_state, global_variances, active_dims)
        votes[factor_index, argmin] += 1
    return votes
