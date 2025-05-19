import numpy as np
from utils.log_utils import _debug, _info


def _generate_key_relationships(first_key, last_key):
    """
    Method to generate key relationships between two keys.
    :param first_key: The first key.
    :param last_key: The last key.
    :return: The key relationships.
    """
    # initial message
    _info("🔄 Pattern generation started...")

    # initialize data
    key_relationships = {}
    keys = list(range(first_key, last_key))

    # for each key
    for i, key in enumerate(keys):

        # initialize related keys to the current one
        related = []

        # relations between the previous and next key
        if i - 1 >= 0:
            related.append(keys[i - 1])
        if i + 1 < len(keys):
            related.append(keys[i + 1])

        # store the related keys
        key_relationships[key] = related

    # show a successful message
    _info(f"🟢 Key relationships generated.")

    return key_relationships


def _generate_access_pattern_requests(
        last_accessed_key,
        key_relationships,
        probs,
        config_settings
):
    """
    Method to generate access pattern requests.
    :param last_accessed_key: The last accessed key.
    :param key_relationships: The key relationships.
    :param probs: The probabilities of the keys.
    :param config_settings: The configuration settings.
    :return: The requested generate, following a specific access pattern.
    """
    # initial message
    _info("🔄 Access pattern requests generation started...")

    if (
            last_accessed_key is None
            or np.random.rand() > config_settings.locality_prob
    ):
        # generate the request following Zipf distribution
        request = np.random.choice(
            np.arange(
                config_settings.first_key,
                config_settings.last_key
            ),
            p=probs
        )
    else:
        # get the related keys
        related_keys = key_relationships.get(
            last_accessed_key,
            []
        )

        if related_keys:
            # generate a request following the relation among keys
            request = np.random.choice(related_keys)
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
    _info(f"🟢 Access pattern requests generated.")

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
    _info("🔄 Temporal access pattern requests generation started...")

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
    _info(f"🟢 Temporal access pattern requests generated.")

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
    _info("🔄 Pattern requests generation started...")

    # debugging
    _debug(f"⚙️ Probabilities length: {len(probs)}.")
    _debug(f"⚙️ Number of requests: {num_requests}.")
    _debug(f"⚙️ Timestamps length: {len(timestamps)}.")

    # initialize data
    requests = []
    delta_times = []
    last_accessed_key = None

    # define the day as period (24(h) * 60 (min) * 60(s))
    period = 24 * 60 * 60

    # debugging
    _debug(f"⚙️ Period: {period}.")

    # check timestamps and probabilities
    if len(timestamps) == 0:
        raise ValueError("❌ timestamps list cannot be empty.")
    if (
        not isinstance(probs, np.ndarray) or
        not np.isclose(np.sum(probs), 1.0)
    ):
        raise ValueError("❌ probs must be a numpy array summing to 1.")

    # define key relationships using a dictionary
    key_relationships = _generate_key_relationships(
        config_settings.first_key,
        config_settings.last_key
    )

    # to make the process deterministic
    np.random.seed(config_settings.seed)

    try:
        # for each request
        for i in range(num_requests):

            # generate a request following a specific
            # access pattern
            request = _generate_access_pattern_requests(
                last_accessed_key,
                key_relationships,
                probs,
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
            _debug(f"⚙️ Number of request generated: {i}.")
            _debug(f"⚙️ Request generated: {request}.")
            _debug(f"⚙️ Delta time generated: {delta_t}.")

    except (ValueError, TypeError, IndexError, ZeroDivisionError,
            AttributeError, MemoryError) as e:
        raise RuntimeError(f"❌ Error while generating data access pattern: {e}.")

    # show a successful message
    _info(f"🟢 Pattern requests generated.")

    return requests, delta_times