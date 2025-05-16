import numpy as np
from data_generation.zipf_calculator import _calculate_zipf_distribution_probs
from utils.log_utils import _info, _debug
from utils.config_utils import _get_config_value


def _show_freq_table(requests):
    """
    Method to show the frequency table of the keys in the dataset.
    :param requests: Requests created.
    :return:
    """
    # get the number of requests per key
    unique, counts = np.unique(
        requests,
        return_counts=True
    )

    # calculate the percentages
    percentages = counts / len(requests) * 100

    # create a table
    table_lines = ["Key | Occurrences | %", "-" * 30]

    for u, c, p in zip(unique, counts, percentages):
        table_lines.append(f"{u:6} | {c:10} | {p:10.2f}%")
    table_str = "\n".join(table_lines)

    # show the created table
    _info("📊 Frequency table:\n%s", table_str)


def _generate_requests_with_delta_times(probs, size=None):
    """
    Method to generate requests with delta times between events.
    :param probs: Zipf's probabilities.
    :return: Requests and delta times between generated events.
    """
    # read configurations
    min_key = _get_config_value("data.first_key")
    max_key = _get_config_value("data.last_key") + 1
    num_requests = _get_config_value("data.num_requests") \
        if size is None else size

    np.random.seed(_get_config_value("data.seed"))

    # generate requests
    requests = np.random.choice(
        np.arange(min_key, max_key),
        size=num_requests,
        p=probs
    )

    # generate timestamps with an average freq
    freq = np.random.exponential(
        scale=_get_config_value("data.freq_timestamp"),
        size=num_requests
    )
    timestamps = np.cumsum(freq).astype(int)

    # calculate the temporal difference between events
    delta_times = np.diff(timestamps, prepend=timestamps[0])

    return requests, delta_times


def _generate_static_requests():
    """
    Method to orchestrate the static requests' generation.
    :return: Static requests and delta times generated.
    """
    # initial message
    _info("🔄 Static requests generation started...")

    # read configurations
    num_requests = _get_config_value("data.num_requests")
    num_keys = _get_config_value("data.num_keys")
    alpha = _get_config_value("data.alpha")
    min_key = _get_config_value("data.first_key")
    max_key = _get_config_value("data.last_key") + 1

    # debugging
    _debug(f"⚙️Zipf parameter: {alpha}.")
    _debug(f"⚙️Number of keys: {num_keys}.")
    _debug(f"⚙️Requests range: ({min_key}, {max_key - 1}).")

    # check the validity of the parameters
    if num_requests <= 0 or num_keys <= 0:
        raise ValueError("❌ num_requests and num_keys must be positive integers.")
    if alpha <= 0:
        raise ValueError("❌ alpha must be a positive float.")

    # calculate the probabilities
    probs = _calculate_zipf_distribution_probs(
        np.arange(min_key, max_key),
        alpha
    )

    # generate requests
    requests, delta_times = _generate_requests_with_delta_times(probs)

    # debugging
    _debug(f"⚙️Requests generated: {len(requests)}.")
    _debug(f"⚙️Delta times shape: {delta_times.shape}.")

    # show a frequency table of the keys
    _show_freq_table(requests)

    # show a successful message
    _info("🟢 Static requests generated.")

    return requests, delta_times


def _generate_dynamic_requests():
    """
    Method to orchestrate the dynamic requests' generation.
    :return: Dynamic requests and delta times generated.
    """
    # initial message
    _info("🔄 Dynamic requests generation started...")

    # read configurations
    num_requests = _get_config_value("data.num_requests")
    num_keys = _get_config_value("data.num_keys")
    alpha_start = _get_config_value("data.alpha_start")
    alpha_end = _get_config_value("data.alpha_end")
    time_steps = _get_config_value("data.time_steps")
    min = _get_config_value("data.first_key")
    max = _get_config_value("data.last_key") + 1

    # debugging
    _debug(f"⚙️Zipf parameters (from-to): ({alpha_start} - {alpha_end}).")
    _debug(f"⚙️Number of keys: {num_keys}.")
    _debug(f"⚙️Requests range: ({min}, {max - 1}).")
    _debug(f"⚙️Total time steps: {time_steps}.")

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

    requests, delta_times = [], []

    # for each alpha value
    for t, alpha in enumerate(alpha_values):
        # calculate the probabilities
        probs = _calculate_zipf_distribution_probs(
            np.arange(min, max),
            alpha
        )

        # generate requests
        reqs, dt = _generate_requests_with_delta_times(
            probs,
            time_step_duration
        )

        requests.extend(reqs)
        delta_times.extend(dt)

    # show a frequency table of the keys
    _show_freq_table(requests)

    # show a successful message
    _info("🟢 Dynamic requests generated.")

    return requests, delta_times