import numpy as np
import logging
from data_generation.zipf_calculator import _calculate_zipf_distribution_probs
from utils.config_utils import _get_config_value


def _generate_requests_with_timestamps(
        num_keys,
        num_requests,
        probs
):
    """
    Method to generate requests with timestamps.
    :param num_keys: Number of keys.
    :param num_requests: Number of requests to generate.
    :param probs: Zipf's probabilities.
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
        scale=_get_config_value("data.freq_timestamp"),
        size=num_requests
    )
    timestamps = np.cumsum(freq).astype(int)

    return requests, timestamps


def _generate_static_requests():
    """
    Method to orchestrate the static requests and timestamps generation.
    :return: Static requests and timestamps generated.
    """
    # initial message
    logging.info("🔄 Static requests generation started...")

    # read configurations
    num_requests = _get_config_value("data.num_requests")
    num_keys = _get_config_value("data.num_keys")
    alpha = _get_config_value("data.alpha")

    # check the validity of the parameters
    if num_requests <= 0 or num_keys <= 0:
        raise ValueError("❌ num_requests and num_keys must be positive integers.")
    if alpha <= 0:
        raise ValueError("❌ alpha must be a positive float.")

    # calculate the probabilities
    probs = _calculate_zipf_distribution_probs(
        np.arange(1, num_keys + 1),
        alpha
    )

    # generate requests with timestamps
    requests, timestamps = _generate_requests_with_timestamps(
        num_keys,
        num_requests,
        probs
    )

    # show a successful message
    logging.info("🟢 Static requests and timestamps generated.")

    return requests, timestamps


def _generate_dynamic_requests():
    """
    Method to orchestrate the dynamic requests and timestamps generation.
    :return: Dynamic requests and timestamps generated.
    """
    # initial message
    logging.info("🔄 Dynamic requests generation started...")

    # read configurations
    num_requests = _get_config_value("data.num_requests")
    num_keys = _get_config_value("data.num_keys")
    alpha_start = _get_config_value("data.alpha_start")
    alpha_end = _get_config_value("data.alpha_end")
    time_steps = _get_config_value("data.time_steps")

    # generate the Zipf distribution's parameter values
    alpha_values = np.linspace(alpha_start, alpha_end, time_steps)

    # check the validity of the parameters
    if num_requests <= 0 or num_keys <= 0 or time_steps <= 0:
        raise ValueError("❌ num_requests, num_keys, and time_steps must be positive integers.")
    if len(alpha_values) != time_steps:
        raise ValueError("❌ alpha_values length must match time_steps.")
    if any(alpha <= 0 for alpha in alpha_values):
        raise ValueError("❌ All alpha values must be positive.")

    # calculate the time step duration
    time_step_duration = num_requests // time_steps

    # check if some requests are ignored
    remainder = num_requests % time_steps
    # show a warning message if some requests are ignored
    if remainder > 0:
        logging.warning(f"⚠️{remainder} requests will be ignored due to uneven split.")

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
            probs
        )

        if timestamps:
            ts += timestamps[-1] + 1

        requests.extend(reqs)
        timestamps.extend(ts)

    # show a successful message
    logging.info("🟢 Dynamic requests and timestamps generated.")

    return requests, timestamps