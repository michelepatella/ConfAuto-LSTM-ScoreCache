from data_generation.generators.patterns.access_pattern_generator import generate_access_pattern
from data_generation.generators.patterns.temporal_pattern_generator import generate_temporal_pattern
from utils.logs.log_utils import debug, info
import numpy as np


def generate_pattern_requests(
        probs,
        num_requests,
        timestamps,
        config_settings
):
    """
    Method to orchestrate requests and timestamps generation
    based on a combination of access pattern and temporal access pattern.
    :param probs: The Zipf probabilities.
    :param num_requests: The number of requests.
    :param timestamps: The list of timestamps.
    :param config_settings: The configuration settings.
    :return: A tuple containing the generated requests and timestamps.
    """
    # initial message
    info("🔄 Pattern requests generation started...")

    # debugging
    debug(f"⚙️ Probabilities length: {len(probs)}.")
    debug(f"⚙️ Number of requests: {num_requests}.")
    debug(f"⚙️ Timestamps length: {len(timestamps)}.")

    # initialize data
    requests = []
    day = 0
    time_in_day = 0.0

    if not timestamps:
        timestamps = [0.0]
    else:
        timestamps = list(timestamps)

    # get the range of all possible keys
    key_range = np.arange(
        config_settings.first_key,
        config_settings.last_key
    )

    # define the day as period
    period = 24 * 60 * 60

    # debugging
    debug(f"⚙️ Period: {period}.")

    # check timestamps and probabilities
    if len(timestamps) == 0:
        raise ValueError("❌ timestamps list cannot be empty.")
    if (
        not isinstance(probs, np.ndarray) or
        not np.isclose(np.sum(probs), 1.0)
    ):
        raise ValueError("❌ probs must be a numpy array summing to 1.")

    try:
        # to make the process deterministic
        np.random.seed(config_settings.seed)

        # for each request
        for i in range(num_requests):
            # generate the delta time
            delta_t = generate_temporal_pattern(
                [timestamps[-1] % period],
                period,
                config_settings
            )

            if time_in_day + delta_t > period:
                day += 1
                time_in_day = (time_in_day + delta_t) - period
            else:
                time_in_day += delta_t
            total_time = day * period + time_in_day

            # generate request
            request = generate_access_pattern(
                probs,
                key_range,
                total_time,
                requests,
                config_settings,
            )

            # store data
            requests.append(request)
            timestamps.append(total_time)

            # debugging
            debug(f"⚙️ Number of request generated: {i+1}.")
            debug(f"⚙️ Request generated: {request}.")
            debug(f"⚙️ Timestamps generated: {timestamps}.")

    except (
            ValueError,
            TypeError,
            IndexError,
            ZeroDivisionError,
            AttributeError,
            MemoryError
    ) as e:
        raise RuntimeError(f"❌ Error while generating data access pattern: {e}.")

    # show a successful message
    info(f"🟢 Pattern requests generated.")

    return requests, timestamps