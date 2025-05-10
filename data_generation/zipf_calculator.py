import numpy as np


def _calculate_zipf_distribution_probs(keys, alpha):
    """
    Method to calculate the Zipf distribution's probabilities.
    :param keys: List of keys.
    :param alpha: Zipf distribution's parameter.
    :return: Zipf distribution's probabilities as output.
    """
    # calculate the probability of the keys according to the Zipf's distribution
    probs = 1.0 / np.power(keys, alpha)

    # normalize probabilities to make sum to 1
    probs = probs / np.sum(probs)

    return probs