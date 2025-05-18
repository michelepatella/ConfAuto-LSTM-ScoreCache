import numpy as np
from config.main import zipf_alpha_start, zipf_alpha_end, zipf_time_steps, num_requests, first_key, last_key, zipf_alpha
from data_generation.pattern_generator import _generate_pattern
from data_generation.zipf_calculator import _calculate_zipf_distribution_probs
from utils.log_utils import _info, _debug


def _show_freq_table(requests):
    """
    Method to show the frequency table of the keys in the dataset.
    :param requests: Requests created.
    :return:
    """
    # initial message
    _info("🔄 Requests frequency table building started...")

    try:
        # get the number of requests per key
        unique, counts = np.unique(
            requests,
            return_counts=True
        )

        # calculate the percentages
        percentages = counts / len(requests) * 100

        # create a table
        table_lines = ["Key | Occurrences | %", "-" * 30]

        # add all lines
        for u, c, p in zip(unique, counts, percentages):
            table_lines.append(f"{u:6} | {c:10} | {p:10.2f}%")
        table_str = "\n".join(table_lines)

    except (TypeError, ValueError, ZeroDivisionError) as e:
        raise RuntimeError(f"❌ Error while building the requests frequency table: {e}.")

    # show a successful message
    _info("🟢 Requests frequency table built.")

    # show the created table
    _info("📊 Frequency table:\n%s", table_str)


def _generate_static_requests():
    """
    Method to orchestrate the static requests' generation.
    :return: Static requests and delta times generated.
    """
    # initial message
    _info("🔄 Static requests generation started...")

    # calculate the probabilities
    probs = _calculate_zipf_distribution_probs(
        np.arange(first_key, last_key),
        zipf_alpha
    )

    timestamps = [0]
    # generate patterns
    requests, delta_times = _generate_pattern(
        probs,
        num_requests,
        timestamps
    )

    # debugging
    _debug(f"⚙️Probabilities length: {len(probs)}.")
    _debug(f"⚙️Requests length: {len(requests)}.")
    _debug(f"⚙️Delta times length: {len(delta_times)}.")

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

    # generate the Zipf distribution's parameter values
    alpha_values = np.linspace(
        zipf_alpha_start,
        zipf_alpha_end,
        zipf_time_steps
    )

    # debugging
    _debug(f"⚙️Alpha values length: {len(alpha_values)}.")

    # check validity of generated alpha values
    if len(alpha_values) != zipf_time_steps:
        raise ValueError("❌ alpha_values length must match time_steps.")
    if any(alpha <= 0 for alpha in alpha_values):
        raise ValueError("❌ All alpha values must be positive.")

    # calculate the time step duration
    time_step_duration = num_requests // zipf_time_steps

    # debugging
    _debug(f"⚙️Time step duration: {time_step_duration}.")

    # initialize data
    requests = []
    delta_times = []
    timestamps = [0]

    # for each alpha value
    for t, alpha in enumerate(alpha_values):

        # calculate the probabilities
        probs = _calculate_zipf_distribution_probs(
            np.arange(first_key, last_key),
            alpha
        )

        # generate pattern
        reqs, dt = _generate_pattern(
            probs,
            time_step_duration,
            timestamps
        )

        # store generated data
        requests.extend(reqs)
        delta_times.extend(dt)

        # debugging
        _debug(f"⚙️Probabilities length: {len(probs)}.")

    # debugging
    _debug(f"⚙️Requests length: {len(requests)}.")
    _debug(f"⚙️Delta times length: {len(delta_times)}.")

    # show a frequency table of the keys
    _show_freq_table(requests)

    # show a successful message
    _info("🟢 Dynamic requests generated.")

    return requests, delta_times