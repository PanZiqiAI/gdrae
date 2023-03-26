""" Disentanglement, Completeness and Informativeness (DCI). """

import scipy
from ..utils import *
from six.moves import range
from sklearn import ensemble
from scipy import stats


def compute_dci(ground_truth_data, representation_function, random_state, num_train=100, num_test=100, batch_size=16):
    """ Computes the DCI scores according to Sec 2.
    :param ground_truth_data: GroundTruthData to be sampled from.
    :param representation_function: Function that takes observations as input and outputs a dim_representation sized
        representation for each observation.
    :param random_state: Numpy random state used for randomness.
    :param num_train: Number of points used for training.
    :param num_test: Number of points used for testing.
    :param batch_size: Batch size for sampling.
    :return: Dictionary with average disentanglement score, completeness and informativeness (train and test).
    """
    # Get mu:(nz, num_train), labels:(num_factors, num_train)
    mus_train, ys_train = generate_batch_factor_code(ground_truth_data, representation_function, num_train, random_state, batch_size)
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    mus_test, ys_test = generate_batch_factor_code(ground_truth_data, representation_function, num_test, random_state, batch_size)
    scores = _compute_dci(mus_train, ys_train, mus_test, ys_test)
    return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """ Computes score based on both training and testing codes and factors. """
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """ Compute importance based on gradient boosted trees. """
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = ensemble.GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
    """ Compute disentanglement score of each code. """
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """ Compute completeness of the representation. """
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)
