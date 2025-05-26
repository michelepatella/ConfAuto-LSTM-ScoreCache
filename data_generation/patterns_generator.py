from utils.log_utils import debug, info
import numpy as np


def _modify_zipf_distribution(probs, current_time, config_settings):
    """
    Method to modify zipf distribution, according to the hour of the day.
    :param probs: The Zipf distribution probabilities.
    :param current_time: The current time.
    :param config_settings: The config settings.
    :return: The key range and modified zipf distribution probabilities.
    """
    # initial message
    info("🔄 Zipf distribution modification started...")

    try:
        # calculate the hour
        day_seconds = 24 * 60 * 60
        hour_of_day = (current_time % day_seconds) / 3600.0

        key_range = np.arange(config_settings.first_key, config_settings.last_key)
        weights = np.ones_like(key_range, dtype=float)

        # modify zipf based on the hour of the day
        if 6 <= hour_of_day < 12:
            # morning: prefer first keys
            weights *= np.exp(-0.2 * (key_range - config_settings.first_key))
        elif 12 <= hour_of_day < 18:
            # afternoon: prefer mid-keys
            mid = (config_settings.first_key + config_settings.last_key) / 2
            weights *= np.exp(-0.01 * (key_range - mid) ** 2)
        elif 18 <= hour_of_day < 24:
            # evening: prefer last keys
            weights *= np.exp(-0.2 * (config_settings.last_key - key_range))
        else:
            # night: distribution is flattened
            weights *= 1.0

        # apply and normalize zipf probabilities
        modified_probs = probs * weights
        modified_probs /= modified_probs.sum()

    except (ZeroDivisionError, TypeError, ValueError) as e:
        raise RuntimeError(f"❌ Error while modifying Zipf distribution: {e}.")

    # show a successful message
    info(f"🟢 Access pattern requests generated.")

    return key_range, modified_probs


def _generate_access_pattern_requests(
        last_accessed_key,
        probs,
        current_time,
        config_settings
):
    """
    Method to generate access pattern requests.
    :param last_accessed_key: The last accessed key.
    :param probs: The probabilities of the keys.
    :param current_time: The current time.
    :param config_settings: The configuration settings.
    :return: The requested generate, following a specific access pattern.
    """
    # initial message
    info("🔄 Access pattern requests generation started...")

    # modify zipf distribution based on the hour of the day
    key_range, modified_probs = _modify_zipf_distribution(
        probs,
        current_time,
        config_settings
    )

    # generate the request following Zipf distribution
    request = np.random.choice(
        key_range,
        p=modified_probs
    )

    # show a successful message
    info(f"🟢 Access pattern requests generated.")

    return request


def _generate_temporal_access_pattern_requests(
        i,
        timestamps,
        period,
        config_settings
):
    """
    Method to generate temporal access pattern requests.
    :param i: The current index.
    :param timestamps: The timestamps.
    :param period: The period.
    :param config_settings: The configuration settings.
    :return: The delta time generated.
    """
    # initial message
    info("🔄 Temporal access pattern requests generation started...")

    # calculate periodic component for frequency scaling
    periodic_scale = (config_settings.periodic_base_scale +
                      config_settings.periodic_amplitude
                      * np.cos(2 * np.pi * timestamps[-1] / period - np.pi))

    # introduce burstiness
    if i % config_settings.burst_every < config_settings.burst_peak:
        bursty_scale = config_settings.burst_high
    else:
        bursty_scale = config_settings.burst_low

    # combine periodic and bursty scales
    freq_scale = periodic_scale * bursty_scale

    # Calculate delta time
    delta_t = np.random.exponential(scale=freq_scale)

    # show a successful message
    info(f"🟢 Temporal access pattern requests generated.")

    return delta_t


def _generate_pattern_requests(
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
    delta_times = []
    last_accessed_key = None

    # define the day as period (24(h) * 60 (min) * 60(s))
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

            # generate a request following a specific
            # access pattern
            request = _generate_access_pattern_requests(
                last_accessed_key,
                probs,
                timestamps[-1],
                config_settings
            )

            # generate a delta time following specific
            # temporal access pattern
            delta_t = _generate_temporal_access_pattern_requests(
                i,
                timestamps,
                period,
                config_settings
            )

            # store generated data
            requests.append(request)
            timestamps.append(timestamps[-1] + delta_t)
            delta_times.append(delta_t)
            last_accessed_key = request

            # debugging
            debug(f"⚙️ Number of request generated: {i+1}.")
            debug(f"⚙️ Request generated: {request}.")
            debug(f"⚙️ Timestamps generated: {timestamps}.")

    except (ValueError, TypeError, IndexError, ZeroDivisionError,
            AttributeError, MemoryError) as e:
        raise RuntimeError(f"❌ Error while generating data access pattern: {e}.")

    # show a successful message
    info(f"🟢 Pattern requests generated.")

    return requests, timestamps