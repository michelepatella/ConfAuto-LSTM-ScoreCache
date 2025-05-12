import numpy as np
import logging
from data_generation.zipf_calculator import _calculate_zipf_distribution_probs
from utils.config_utils import load_config, get_config_value


def _generate_requests_with_timestamps(
        num_keys,
        num_requests,
        probs,
        config
):
    """
    Method to generate requests with timestamps.
    :param num_keys: Number of keys.
    :param num_requests: Number of requests to generate.
    :param probs: Zipf's probabilities.
    :param config: Config object.
    :return: Requests and timestamps generated.
    """
    # generate requests
    requests = np.random.choice(
        np.arange(1, num_keys + 1),
        size=num_requests,
        p=probs
    )

    # generate timestamps with an average freq
    freq = np.random.exponential(
        scale=get_config_value(config, "data.freq_timestamp"),
        size=num_requests
    )
    timestamps = np.cumsum(freq).astype(int)

    return requests, timestamps


def _generate_static_requests(config):
    """
    Method to generate static requests (with the associated timestamps).
    :param config: Config object.
    :return: Static requests and timestamps.
    """
    # read configurations
    num_requests = get_config_value(config, "data.num_requests")
    num_keys = get_config_value(config, "data.num_keys")
    alpha = get_config_value(config, "data.alpha")

    # check the validity of the parameters
    if num_requests <= 0 or num_keys <= 0:
        raise ValueError("num_requests and num_keys must be positive integers.")
    if alpha <= 0:
        raise ValueError("alpha must be a positive float.")

    # calculate the probabilities
    probs = _calculate_zipf_distribution_probs(
        np.arange(1, num_keys + 1),
        alpha
    )

    # generate requests with timestamps
    requests, timestamps = _generate_requests_with_timestamps(
        num_keys,
        num_requests,
        probs,
        config
    )

    # show a successful message
    logging.info(f"🟢Static requests and timestamps generated.")

    return requests, timestamps


def _generate_dynamic_requests(config):
    """
    Method to generate dynamic requests (with the associated timestamps).
    :param config: Config object.
    :return: Dynamic requests and timestamps.
    """
    # read configurations
    num_requests = get_config_value(config, "data.num_requests")
    num_keys = get_config_value(config, "data.num_keys")
    alpha_start = get_config_value(config, "data.alpha_start")
    alpha_end = get_config_value(config, "data.alpha_end")
    time_steps = get_config_value(config, "data.time_steps")

    # generate the Zipf distribution's parameter values
    alpha_values = np.linspace(alpha_start, alpha_end, time_steps)

    # check the validity of the parameters
    if num_requests <= 0 or num_keys <= 0 or time_steps <= 0:
        raise ValueError("num_requests, num_keys, and time_steps must be positive integers.")
    if len(alpha_values) != time_steps:
        raise ValueError("alpha_values length must match time_steps.")
    if any(alpha <= 0 for alpha in alpha_values):
        raise ValueError("All alpha values must be positive.")

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

        # generate requests with timestamps
        reqs, ts = _generate_requests_with_timestamps(
            num_keys,
            time_step_duration,
            probs,
            config
        )

        if timestamps:
            ts += timestamps[-1] + 1

        requests.extend(reqs)
        timestamps.extend(ts)

    # show a successful message
    logging.info(f"🟢Dynamic requests and timestamps generated.")

    return requests, timestamps