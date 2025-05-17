import numpy as np
from config.main import first_key, last_key, periodic_amplitude, burst_every, burst_peak, \
    burst_high, burst_low, periodic_base_scale
from utils.log_utils import _debug, _info


def _generate_key_relationships(first_key, last_key):
    key_relationships = {}
    keys = list(range(first_key, last_key + 1))

    for i, key in enumerate(keys):
        related = []

        if i - 1 >= 0:
            related.append(keys[i - 1])
        if i + 1 < len(keys):
            related.append(keys[i + 1])

        key_relationships[key] = related

    return key_relationships


def _generate_pattern(probs, num_requests, timestamps):
    """
    Generates requests and delta times based on a combination of bursty and periodic pattern,
    with added key relationship logic.

    :param probs: The Zipf probabilities.
    :param num_requests: The number of requests.
    :param timestamps: The list of timestamps.
    :return: A tuple containing the generated requests and delta_times.
    """
    # initial message
    _info("🔄 Pattern generation started...")

    # debugging
    _debug(f"⚙️ Probabilities length: {len(probs)}.")
    _debug(f"⚙️ Number of requests: {num_requests}.")
    _debug(f"⚙️ Timestamps length: {len(timestamps)}.")

    # initialize data
    requests = []
    delta_times = []
    period = 24 * 60 * 60  # Define the day as period (24(h) * 60 (min) * 60(s))

    # debugging
    _debug(f"⚙️ Period: {period}.")

    # check timestamps and probabilities
    if len(timestamps) == 0:
        raise ValueError("❌ timestamps list cannot be empty.")
    if not isinstance(probs, np.ndarray) or not np.isclose(np.sum(probs), 1.0):
        raise ValueError("❌ probs must be a numpy array summing to 1.")

    # define key relationships using a dictionary
    key_relationships = _generate_key_relationships(first_key, last_key)

    last_accessed_key = None
    print(len(probs))
    print(len(np.arange(first_key, last_key + 1)))
    try:
        for i in range(num_requests):
            if last_accessed_key is None:
                # generate the first request
                request = np.random.choice(
                    np.arange(first_key, last_key + 1),
                    p=probs
                )
            else:
                 #next request based on relations
                related_keys = key_relationships.get(last_accessed_key, [])
                if related_keys:
                    request = np.random.choice(related_keys)
                else:
                    request = np.random.choice(
                        np.arange(first_key, last_key + 1),
                        p=probs
                    )

            # calculate periodic component for frequency scaling
            periodic_scale = (
                5 + 5 * np.cos(2 * np.pi * timestamps[-1] / period)
            )

            # Introduce burstiness
            if i % burst_every < burst_peak:
                bursty_scale = burst_high
            else:
                bursty_scale = burst_low

            # Combine periodic and bursty scales
            freq_scale = periodic_scale * bursty_scale

            # Calculate delta time
            delta_t = np.random.exponential(scale=freq_scale)

            # store generated data
            requests.append(request)
            timestamps.append(timestamps[-1] + delta_t)
            delta_times.append(delta_t)
            last_accessed_key = request #update

            # debugging
            _debug(f"⚙️ Number of request generated: {i}.")
            _debug(f"⚙️ Frequency scale calculated: {freq_scale}.")
            _debug(f"⚙️ Request generated: {request}.")
            _debug(f"⚙️ Delta time generated: {delta_t}.")

    except (ValueError, TypeError, IndexError, ZeroDivisionError,
            AttributeError, MemoryError) as e:
        raise RuntimeError(f"❌ Error while generating data access pattern: {e}.")

    # show a successful message
    _info(f"🟢 Pattern generated.")

    return requests, delta_times