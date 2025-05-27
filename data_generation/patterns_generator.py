from utils.log_utils import debug, info
import numpy as np


def _generate_access_pattern_requests(
        probs,
        key_range,
        current_time,
        history_keys,
        config_settings
):
    """
    Method to generate access pattern requests.
    :param probs: The probabilities of the keys.
    :param key_range: The range of possible keys.
    :param current_time: The current time.
    :param history_keys: The history keys.
    :param config_settings: The configuration settings.
    :return: The requested generate, following a specific access pattern.
    """
    # initial message
    info("🔄 Access pattern requests generation started...")

    hour = current_time % 24
    base = config_settings.first_key  # 1
    key_blocks = {
        (0, 4): list(range(base, base + 5)),  # 1-5
        (4, 8): list(range(base + 5, base + 10)),  # 6-10
        (8, 12): list(range(base + 10, base + 15)),  # 11-15
        (12, 16): list(range(base + 15, base + 20)),  # 16-20
        (16, 20): list(range(base + 20, base + 25)),  # 21-25
        (20, 24): list(range(base + 25, base + 30))  # 26-30
    }

    for (start, end), keys in key_blocks.items():
        if start <= hour < end:
            if (start, end) == (0, 4):
                idx = len(history_keys) % len(keys)
                return keys[idx]
            elif (start, end) == (4, 8):
                return np.random.choice(keys, p=np.ones(len(keys)) / len(keys))
            elif (start, end) == (8, 12):
                if len(history_keys) % 2 == 0:
                    return keys[::2][len(history_keys) % len(keys[::2])]
                else:
                    return keys[1::2][len(history_keys) % len(keys[1::2])]
            elif (start, end) == (12, 16):
                cycle = [keys[0], keys[2], keys[4]]
                return cycle[len(history_keys) % len(cycle)]
            elif (start, end) == (16, 20):
                if len(history_keys) % 3 == 0:
                    return keys[len(history_keys) % len(keys)]
                else:
                    return np.random.choice(keys)
            elif (start, end) == (20, 24):
                return np.random.choice(key_range, p=probs)

    # show a successful message
    info(f"🟢 Access pattern requests generated.")

    # Default fallback
    return np.random.choice(key_range, p=probs)


def _generate_temporal_access_pattern_requests(
        timestamps,
        period,
        config_settings
):
    """
    Method to generate temporal access pattern requests.
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
                      * np.cos(2 * np.pi * timestamps[-1]))

    # extract hour of the day from timestamp
    hour_of_day = (timestamps[-1] % period)

    # generate burst middle of the day
    if config_settings.burst_hour_start <= hour_of_day <= config_settings.burst_hour_end:
        bursty_scale = config_settings.burst_high
    else:
        bursty_scale = config_settings.burst_low

    # combine periodic and bursty scales
    freq_scale = periodic_scale * bursty_scale

    # calculate delta time
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
    timestamps = [0.0]
    key_range = np.arange(
        config_settings.first_key,
        config_settings.last_key
    )

    # define the day as period
    period = 24

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
                probs,
                key_range,
                timestamps[-1],
                requests,
                config_settings,
            )

            # generate a delta time following specific
            # temporal access pattern
            delta_t = _generate_temporal_access_pattern_requests(
                timestamps,
                period,
                config_settings
            )

            # store generated data
            requests.append(request)
            timestamps.append(timestamps[-1] + delta_t)
            delta_times.append(delta_t)

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