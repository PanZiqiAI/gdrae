""" SAP score. """

from ..utils import *
from sklearn import svm
from six.moves import range


def compute_sap(ground_truth_data, representation_function, random_state, num_train=100, num_test=100, batch_size=16, continuous_factors=False):
    """ Computes the SAP score.
    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
        representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param num_train: Number of points used for training.
    :param num_test: Number of points used for testing discrete variables.
    :param batch_size: Batch size for sampling.
    :param continuous_factors: Factors are continuous variable (True) or not (False).
    :return: Dictionary with SAP score.
    """
    mus, ys = generate_batch_factor_code(ground_truth_data, representation_function, num_train, random_state, batch_size)
    mus_test, ys_test = generate_batch_factor_code(ground_truth_data, representation_function, num_test, random_state, batch_size)
    return _compute_sap(mus, ys, mus_test, ys_test, continuous_factors)


def _compute_sap(mus, ys, mus_test, ys_test, continuous_factors):
    """ Computes score based on both training and testing codes and factors. """
    score_matrix = compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors)
    # Score matrix should have shape [num_latents, num_factors].
    assert score_matrix.shape[0] == mus.shape[0]
    assert score_matrix.shape[1] == ys.shape[0]
    scores_dict = {"SAP_score": compute_avg_diff_top_two(score_matrix)}
    return scores_dict


def compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors):
    """ Compute score matrix as described in Section 3. """
    num_latents = mus.shape[0]
    num_factors = ys.shape[0]
    score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):
            mu_i = mus[i, :]
            y_j = ys[j, :]
            if continuous_factors:
                # Attribute is considered continuous.
                cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
                cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
                var_mu = cov_mu_i_y_j[0, 0]
                var_y = cov_mu_i_y_j[1, 1]
                if var_mu > 1e-12:
                    score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
                else:
                    score_matrix[i, j] = 0.
            else:
                # Attribute is considered discrete.
                mu_i_test = mus_test[i, :]
                y_j_test = ys_test[j, :]
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(mu_i[:, np.newaxis], y_j)
                pred = classifier.predict(mu_i_test[:, np.newaxis])
                score_matrix[i, j] = np.mean(pred == y_j_test)
    return score_matrix


def compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])
