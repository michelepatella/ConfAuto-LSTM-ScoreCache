import numpy as np
from data_generation.patterns_generator import _generate_pattern_requests
from data_generation.zipf_calculator import _calculate_zipf_distribution_probs
from utils.log_utils import info, debug


def _generate_static_requests(config_settings):
    """
    Method to orchestrate the static requests' generation.
    :param config_settings: The configuration settings.
    :return: Static requests and delta times generated.
    """
    # initial message
    info("🔄 Static requests generation started...")

    # calculate the probabilities
    probs = _calculate_zipf_distribution_probs(
        np.arange(
            config_settings.first_key,
            config_settings.last_key
        ),
        config_settings.zipf_alpha
    )

    timestamps = [0]
    # generate patterns
    requests, delta_times = _generate_pattern_requests(
        probs,
        config_settings.num_requests,
        timestamps,
        config_settings
    )

    # debugging
    debug(f"⚙️Probabilities length: {len(probs)}.")
    debug(f"⚙️Requests length: {len(requests)}.")
    debug(f"⚙️Delta times length: {len(delta_times)}.")

    # show a successful message
    info("🟢 Static requests generated.")

    return requests, delta_times


def _generate_dynamic_requests(config_settings):
    """
    Method to orchestrate the dynamic requests' generation.
    :param config_settings: The configuration settings.
    :return: Dynamic requests and delta times generated.
    """
    # initial message
    info("🔄 Dynamic requests generation started...")

    # generate the Zipf distribution's parameter values
    alpha_values = np.linspace(
        config_settings.zipf_alpha_start,
        config_settings.zipf_alpha_end,
        config_settings.zipf_time_steps
    )

    # debugging
    debug(f"⚙️Alpha values length: {len(alpha_values)}.")

    # check validity of generated alpha values
    if len(alpha_values) != config_settings.zipf_time_steps:
        raise ValueError("❌ alpha_values length must match time_steps.")
    if any(alpha <= 0 for alpha in alpha_values):
        raise ValueError("❌ All alpha values must be positive.")

    # calculate the time step duration
    time_step_duration = config_settings.num_requests // config_settings.zipf_time_steps

    # debugging
    debug(f"⚙️Time step duration: {time_step_duration}.")

    # initialize data
    requests = []
    delta_times = []
    timestamps = [0]

    # for each alpha value
    for t, alpha in enumerate(alpha_values):

        # calculate the probabilities
        probs = _calculate_zipf_distribution_probs(
            np.arange(
                config_settings.first_key,
                config_settings.last_key
            ),
            alpha
        )

        # generate pattern
        reqs, dt = _generate_pattern_requests(
            probs,
            time_step_duration,
            timestamps,
            config_settings
        )

        # store generated data
        requests.extend(reqs)
        delta_times.extend(dt)

        # debugging
        debug(f"⚙️Probabilities length: {len(probs)}.")

    # debugging
    debug(f"⚙️Requests length: {len(requests)}.")
    debug(f"⚙️Delta times length: {len(delta_times)}.")

    # show a successful message
    info("🟢 Dynamic requests generated.")

    return requests, delta_times