import numpy as np
from utils.log_utils import debug, info


def _generate_key_relationships(first_key, last_key):
    """
    Method to generate key relationships between two keys.
    :param first_key: The first key.
    :param last_key: The last key.
    :return: The key relationships.
    """
    # initial message
    info("🔄 Pattern generation started...")

    key_relationships = {}
    keys = list(range(first_key, last_key))
    num_keys = len(keys)

    for i, key in enumerate(keys):
        prev_key = keys[(i - 1) % num_keys]
        next_key = keys[(i + 1) % num_keys]
        key_relationships[key] = [prev_key, next_key]

    info(f"🟢 Key relationships generated.")
    return key_relationships


def _generate_access_pattern_requests(
        last_accessed_key,
        key_relationships,
        probs,
        current_time,
        config_settings
):
    """
    Method to generate access pattern requests.
    :param last_accessed_key: The last accessed key.
    :param key_relationships: The key relationships.
    :param probs: The probabilities of the keys.
    :param current_time: The current time.
    :param config_settings: The configuration settings.
    :return: The requested generate, following a specific access pattern.
    """
    # initial message
    info("🔄 Access pattern requests generation started...")

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

    if (
            last_accessed_key is None
            or np.random.rand() > config_settings.locality_prob
    ):
        # generate the request following Zipf distribution
        request = np.random.choice(
            key_range,
            p=modified_probs
        )
    else:
        # get the related keys
        related_keys = key_relationships.get(
            last_accessed_key,
            []
        )

        if related_keys:
            # calculate the weights for neighboring keys
            weights = []
            for k in related_keys:
                if k > last_accessed_key:
                    weights.append(config_settings.locality_prob_succ_key)
                else:
                    weights.append(1.0 - config_settings.locality_prob_succ_key)

            weights = np.array(weights)
            weights /= weights.sum()

            request = np.random.choice(related_keys, p=weights)
        else:
            # generate the request following Zipf distribution
            request = np.random.choice(
                np.arange(
                    config_settings.first_key,
                    config_settings.last_key
                ),
                p=probs
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
                      * np.cos(2 * np.pi * timestamps[-1] / period))

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
    Method to orchestrate requests and delta times generation
    based on a combination of access pattern and temporal access pattern.
    :param probs: The Zipf probabilities.
    :param num_requests: The number of requests.
    :param timestamps: The list of timestamps.
    :param config_settings: The configuration settings.
    :return: A tuple containing the generated requests and delta_times.
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
        # define key relationships using a dictionary
        key_relationships = _generate_key_relationships(
            config_settings.first_key,
            config_settings.last_key
        )

        # to make the process deterministic
        np.random.seed(config_settings.seed)

        current_time = timestamps[-1]

        # for each request
        for i in range(num_requests):

            # generate a request following a specific
            # access pattern
            request = _generate_access_pattern_requests(
                last_accessed_key,
                key_relationships,
                probs,
                current_time,
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

            current_time = timestamps[-1]

            # debugging
            debug(f"⚙️ Number of request generated: {i}.")
            debug(f"⚙️ Request generated: {request}.")
            debug(f"⚙️ Delta time generated: {delta_t}.")

    except (ValueError, TypeError, IndexError, ZeroDivisionError,
            AttributeError, MemoryError) as e:
        raise RuntimeError(f"❌ Error while generating data access pattern: {e}.")

    # show a successful message
    info(f"🟢 Pattern requests generated.")

    return requests, delta_times