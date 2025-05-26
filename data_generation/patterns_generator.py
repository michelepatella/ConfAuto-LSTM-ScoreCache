from utils.log_utils import debug, info
import numpy as np


def _modify_zipf_distribution(
        probs,
        period,
        current_time,
        config_settings,
        history_keys=None
):
    """
    Creates a smooth time-based access pattern using harmonics and gradual
    transitions between day phases. Suited for LSTM-like models.
    """
    info("🔄 LSTM-focused time-based distribution generation started...")

    key_range = np.arange(config_settings.first_key, config_settings.last_key)
    total_keys = len(key_range)

    # Normalize time over a 24-hour day [0, 1]
    t = (current_time % period) / period

    # Harmonic time functions to simulate smooth daily transitions
    harmonic1 = np.sin(2 * np.pi * t)              # 1 cycle per day
    harmonic2 = np.sin(4 * np.pi * t + np.pi / 3)  # 2 cycles per day
    harmonic3 = np.cos(6 * np.pi * t + np.pi / 5)  # 3 cycles per day

    # Combine harmonics to calculate dynamic peak position
    dynamic_peak_pos = 0.5 + 0.2 * harmonic1 + 0.15 * harmonic2 + 0.1 * harmonic3
    dynamic_peak_pos = np.clip(dynamic_peak_pos, 0, 1)

    preferred_key_index = config_settings.first_key + dynamic_peak_pos * (total_keys - 1)

    # Optionally bias based on recent history
    if history_keys and len(history_keys) >= 5:
        past_mean = np.mean(history_keys[-5:])
        preferred_key_index = 0.8 * preferred_key_index + 0.6 * past_mean

    # Apply Gaussian bias centered on the preferred index
    sigma = total_keys * 0.03  # peak sharpness
    weights = np.exp(-0.5 * ((key_range - preferred_key_index) / sigma) ** 2) + 1e-6
    weights = weights * 100 + 0.1
    modified_probs = probs * weights
    modified_probs /= modified_probs.sum()

    info("🟢 Time-based access pattern calculated.")
    return key_range, modified_probs


def _generate_access_pattern_requests(
        probs,
        period,
        current_time,
        config_settings,
        history_keys
):
    """
    Method to generate access pattern requests.
    :param probs: The probabilities of the keys.
    :param period: The predefined period.
    :param current_time: The current time.
    :param config_settings: The configuration settings.
    :param history_keys: The history keys.
    :return: The requested generate, following a specific access pattern.
    """
    # initial message
    info("🔄 Access pattern requests generation started...")

    # modify zipf distribution based on the hour of the day
    key_range, modified_probs = _modify_zipf_distribution(
        probs,
        period,
        current_time,
        config_settings,
        history_keys=history_keys
    )

    # generate the request following modified Zipf distribution
    request = np.random.choice(
        key_range,
        p=modified_probs
    )

    # show a successful message
    info(f"🟢 Access pattern requests generated.")

    return request


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
    history_keys = []

    # define the day as period (24(h) * 60 (min) * 60(s))
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
                period,
                timestamps[-1],
                config_settings,
                history_keys=history_keys
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
            history_keys.append(request)

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