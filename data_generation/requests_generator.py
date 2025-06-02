import numpy as np
from data_generation.patterns_generator import _generate_pattern_requests
from data_generation.zipf_calculator import _calculate_zipf_distribution_probs
from utils.log_utils import info, debug


def _timestamps_to_hours(timestamps):
    """
    Method to convert timestamps from
    seconds to hours of the day.
    :param timestamps: Timestamps to convert.
    :return: Timestamps as hours of the day.
    """
    # initial message
    info("🔄 Timestamp convertion started...")

    try:
        # consider timestamps as hours of the day
        timestamps = np.array(timestamps) / 3600.0
    except (
        NameError,
        TypeError,
        ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while converting timestamps to hours of the day: {e}.")

    # show a successful message
    info("🟢 Timestamps converted.")

    return timestamps


def _generate_static_requests(config_settings):
    """
    Method to orchestrate the static requests' generation.
    :param config_settings: The configuration settings.
    :return: Static requests and timestamps generated.
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
    requests, timestamps = _generate_pattern_requests(
        probs,
        config_settings.num_requests,
        timestamps,
        config_settings
    )

    timestamps = _timestamps_to_hours(timestamps)

    # debugging
    debug(f"⚙️Probabilities length: {len(probs)}.")
    debug(f"⚙️Requests length: {len(requests)}.")
    debug(f"⚙️Timestamps length: {len(timestamps)}.")

    # show a successful message
    info("🟢 Static requests generated.")

    return requests, timestamps


def _generate_dynamic_requests(config_settings):
    """
    Method to orchestrate the dynamic requests' generation.
    :param config_settings: The configuration settings.
    :return: Dynamic requests and timestamps generated.
    """
    # initial message
    info("🔄 Dynamic requests generation started...")

    try:
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
        time_step_duration = (
                config_settings.num_requests // config_settings.zipf_time_steps
        )

        # debugging
        debug(f"⚙️Time step duration: {time_step_duration}.")

        # initialize data
        requests = []
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
            reqs, timestamps = _generate_pattern_requests(
                probs,
                time_step_duration,
                timestamps,
                config_settings
            )

            # store generated data
            requests.extend(reqs)

            # debugging
            debug(f"⚙️Probabilities length: {len(probs)}.")

        timestamps = _timestamps_to_hours(timestamps)

        # debugging
        debug(f"⚙️Requests length: {len(requests)}.")
        debug(f"⚙️timestamps length: {len(timestamps)}.")

    except (
        AttributeError,
        TypeError,
        ValueError,
        ZeroDivisionError
    ) as e:
        raise RuntimeError(f"❌ Error while generating dynamic data: {e}.")

    # show a successful message
    info("🟢 Dynamic requests generated.")

    return requests, timestamps