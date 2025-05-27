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


    hour = (current_time / 3600.0) % 24
    base = config_settings.first_key
    keys = list(range(base, config_settings.last_key))
    n_keys = len(keys)
    range_size = config_settings.last_key - config_settings.first_key  # dimensione

    if len(history_keys) < 5:
        return np.random.choice(keys)

    idx = len(history_keys)

    if 0 <= hour < 6:
        if idx % 7 == 0:
            return history_keys[-3]
        return np.random.choice(keys[:n_keys // 3])

    elif 6 <= hour < 12:
        toggle = (idx // 10) % 2
        if toggle == 0:
            # increment modulo corretto
            new_key = ((history_keys[-1] - base + 1) % range_size) + base
            return new_key
        else:
            # decrement modulo corretto
            new_key = ((history_keys[-2] - base - 1) % range_size) + base
            return new_key

    elif 12 <= hour < 18:
        cycle_length = 5 + (idx // 50) % 3
        cycle = keys[:cycle_length]
        return cycle[idx % cycle_length]

    elif 18 <= hour < 22:
        if idx % 4 == 0:
            new_key = ((history_keys[-4] - base + 2) % range_size) + base
            return new_key
        else:
            noise = np.random.randint(-2, 3)
            new_key = ((history_keys[-1] - base + noise) % range_size) + base
            return new_key

    else:
        if idx % 9 == 0:
            return history_keys[-6]
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

    # extract hour of the day
    hour_of_day = (timestamps[-1] % period) / 3600

    # generate a periodic component
    periodic_component = (
            config_settings.periodic_base_scale +
            config_settings.periodic_amplitude *
            np.cos(2 * np.pi * (hour_of_day / 24))
    )

    # generate burst middle of the day
    if config_settings.burst_hour_start <= hour_of_day <= config_settings.burst_hour_end:
        bursty_scale = config_settings.burst_high
    else:
        bursty_scale = config_settings.burst_low

    # combine periodic and bursty scales
    freq_scale = max(0.5, periodic_component * bursty_scale)

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
    day = 0
    time_in_day = 0.0
    timestamps = [0.0]
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
            delta_t = _generate_temporal_access_pattern_requests(
                [timestamps[-1] % period],
                period,
                config_settings
            )

            if time_in_day + delta_t > period:
                # next day
                day += 1
                time_in_day = 0

            time_in_day += delta_t
            total_time = day * period + time_in_day

            # generate request
            request = _generate_access_pattern_requests(
                probs,
                key_range,
                total_time,
                requests,
                config_settings,
            )

            requests.append(request)
            timestamps.append(total_time)

            # debugging
            debug(f"⚙️ Number of request generated: {i+1}.")
            debug(f"⚙️ Request generated: {request}.")
            debug(f"⚙️ Timestamps generated: {timestamps}.")

    except (ValueError, TypeError, IndexError, ZeroDivisionError,
            AttributeError, MemoryError) as e:
        raise RuntimeError(f"❌ Error while generating data access pattern: {e}.")

    # show a successful message
    info(f"🟢 Pattern requests generated.")

    # zip requests e timestamps insieme
    data = list(zip(timestamps, requests))

    # ordina per giorno e poi per ora all'interno del giorno
    data.sort(key=lambda x: (int(x[0] // period), x[0] % period))

    # stampa ordinato
    for ts, req in data:
        day = int(ts // period)
        hour = (ts % period) / 3600
        print(f"Day {day}, Time {hour:.2f}h, Key: {req}")

    return requests, timestamps