import numpy as np
from data_generation.generators.patterns.request_patterns_generator import generate_pattern_requests
from data_generation.utils.converter import timestamps_to_hours
from data_generation.utils.zipf_calculator import calculate_zipf_probs
from utils.logs.log_utils import info, debug


def generate_dynamic_requests(config_settings):
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
            probs = calculate_zipf_probs(
                np.arange(
                    config_settings.first_key,
                    config_settings.last_key
                ),
                alpha
            )

            # generate pattern
            reqs, timestamps = generate_pattern_requests(
                probs,
                time_step_duration,
                timestamps,
                config_settings
            )

            # store generated data
            requests.extend(reqs)

            # debugging
            debug(f"⚙️Probabilities length: {len(probs)}.")

        timestamps = timestamps_to_hours(timestamps)

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