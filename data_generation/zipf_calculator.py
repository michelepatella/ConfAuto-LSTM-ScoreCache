import numpy as np


def _calculate_zipf_distribution_probs(keys, alpha):
    """
    Method to calculate the Zipf distribution's probabilities.
    :param keys: List of keys.
    :param alpha: Zipf distribution's parameter.
    :return: Zipf distribution's probabilities.
    """
    # convert keys to a numpy array for safety
    keys = np.array(keys, dtype=np.float64)

    # check the keys
    if np.any(keys <= 0):
        raise ValueError("All keys must be positive numbers for Zipf distribution.")

    try:
        # calculate the probability of the keys according to the Zipf's distribution
        probs = 1.0 / np.power(keys, alpha)
    except ZeroDivisionError:
        raise ZeroDivisionError("Zipf distribution probabilities cannot be calculated.")

    # normalize probabilities to make sum to 1
    probs = probs / np.sum(probs)

    return probs