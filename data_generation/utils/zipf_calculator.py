import numpy as np
from utils.logs.log_utils import info, debug


def calculate_zipf_probs(
        keys,
        alpha
):
    """
    Method to calculate the Zipf distribution's probabilities.
    :param keys: List of keys.
    :param alpha: Zipf distribution's parameter.
    :return: Zipf distribution's probabilities calculated.
    """
    # initial message
    info("🔄 Zipf distribution's probabilities calculation started...")

    # debugging
    debug(f"⚙️Keys length: {len(keys)}.")
    debug(f"⚙️Alpha value: {alpha}.")

    try:
        # check the keys
        keys = np.array(
            keys,
            dtype=np.float64
        )
    except TypeError as e:
        raise TypeError(f"TypeError: {e}")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}")

    if keys.size == 0:
        raise ValueError(" Keys must be non-empty.")
    if np.any(keys <= 0):
        raise ValueError(" All keys must be strictly positive for Zipf distribution.")

    try:
        # calculate the probability of the keys according to the Zipf's distribution
        probs = 1.0 / np.power(keys, alpha)

        # normalize probabilities to make sum to 1
        probs = probs / np.sum(probs)
    except ZeroDivisionError:
        raise ZeroDivisionError(" Zipf distribution probabilities cannot be calculated due to zero division.")

    # show a successful message
    info("🟢 Zipf distribution's probabilities calculated.")

    return probs