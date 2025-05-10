import numpy as np
import logging
from data_generation.zipf_calculator import _calculate_zipf_distribution_probs
from utils.config_loader import load_config


def _generate_static_requests(num_requests, num_keys, alpha):
    """
    Method to generate static requests (with the associated timestamps).
    :param num_requests: Total number of requests.
    :param num_keys: Total number of keys.
    :param alpha: Zipf distribution's parameter.
    :return: Static requests and timestamps as output.
    """
    # check the validity of the parameters
    if num_requests <= 0 or num_keys <= 0:
        raise ValueError("num_requests and num_keys must be positive integers.")
    if alpha <= 0:
        raise ValueError("alpha must be a positive float.")

    # load config file
    config = load_config()

    # load data config
    if config is not None and "data" in config:
        data_config = config["data"]
    else:
        raise ValueError("Error while loading or reading config file.")

    # calculate the probabilities
    probs = _calculate_zipf_distribution_probs(
        np.arange(1, num_keys + 1),
        alpha
    )

    # generate requests
    requests = np.random.choice(
        np.arange(1, num_keys + 1),
        size=num_requests,
        p=probs
    )

    # generate timestamp randomly, with an average freq
    freq = np.random.exponential(
        scale=data_config['freq_timestamp'] + 1,
        size=num_requests
    )
    timestamps = np.cumsum(freq).astype(int)

    return requests, timestamps

def _generate_dynamic_requests(num_requests, num_keys, alpha_values, time_steps):
    """
    Method to generate dynamic requests (with the associated timestamps).
    :param num_requests: Total number of requests.
    :param num_keys: Total number of keys.
    :param alpha_values: Zipf distribution's parameter values.
    :param time_steps: Total number of time steps.
    :return: Dynamic requests and timestamps as output.
    """
    # check the validity of the parameters
    if num_requests <= 0 or num_keys <= 0 or time_steps <= 0:
        raise ValueError("num_requests, num_keys, and time_steps must be positive integers.")
    if len(alpha_values) != time_steps:
        raise ValueError("alpha_values length must match time_steps.")
    if any(alpha <= 0 for alpha in alpha_values):
        raise ValueError("All alpha values must be positive.")

    # load config file
    config = load_config()

    # load data config
    if config is not None and "data" in config:
        data_config = config["data"]
    else:
        raise ValueError("Error while loading or reading config file.")

    # calculate the time step duration
    time_step_duration = num_requests // time_steps

    # check if some requests are ignored
    remainder = num_requests % time_steps
    # show a warning message if some requests are ignored
    if remainder > 0:
        logging.warning(f"{remainder} requests will be ignored due to uneven split.")

    requests, timestamps = [], []

    # for each alpha value
    for t, alpha in enumerate(alpha_values):
        # calculate the probabilities
        probs = _calculate_zipf_distribution_probs(
            np.arange(1, num_keys + 1),
            alpha
        )

        # generate requests
        reqs = np.random.choice(
            np.arange(1, num_keys + 1),
            size=time_step_duration,
            p=probs
        )
        requests.extend(reqs)

        # generate timestamp randomly, with an average freq
        freq = np.random.exponential(
            scale=data_config['freq_timestamp'] + 1,
            size=time_step_duration
        )
        ts = np.cumsum(freq).astype(int)

        if timestamps:
            ts += timestamps[-1] + 1

        timestamps.extend(ts)

    return requests, timestamps