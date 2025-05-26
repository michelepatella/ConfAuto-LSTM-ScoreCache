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
    Distribuzione modificata semplificata e bilanciata:
    - Evita concentrazione su chiavi basse
    - Introduce variazione dinamica ma su tutto il range
    - Favorisce apprendimento LSTM (nessuna classe è penalizzata)
    """

    # Ora in formato decimale (0-24h)
    hour_of_day = (current_time % period) / 3600.0

    # Range chiavi
    key_range = np.arange(config_settings.first_key, config_settings.last_key)
    total_keys = len(key_range)

    # Centro preferito che varia sinusoidalmente durante il giorno
    # (scorre su tutto il range in 24 ore)
    normalized_hour = hour_of_day / 24.0  # ∈ [0,1]
    preferred_key_index = config_settings.first_key + normalized_hour * (total_keys - 1)

    # Aggiunta di una piccola componente storica per dare coerenza sequenziale
    if history_keys and len(history_keys) >= 3:
        past_mean = np.mean(history_keys[-3:])
        preferred_key_index = (preferred_key_index + past_mean) / 2

    # Deviazione fissa per includere più chiavi nel focus
    sigma = total_keys * 0.2  # più larga per coprire più classi

    # Costruzione dei pesi con gaussiana
    weights = np.exp(-0.5 * ((key_range - preferred_key_index) / sigma) ** 2)

    # Evita che pesi troppo piccoli spariscano (aiuta le classi meno frequenti)
    weights += 1e-6

    # Applicazione dei pesi
    modified_probs = probs * weights
    modified_probs /= modified_probs.sum()

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
    hour_of_day = (timestamps[-1] % period) / 3600.0

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