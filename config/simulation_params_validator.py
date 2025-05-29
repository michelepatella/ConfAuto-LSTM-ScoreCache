from config.config_utils import get_config_value
from utils.log_utils import info


def _check_simulation_general_params(cache_size):
    """
    Method to check simulation general parameters.
    :param cache_size: The size of the cache.
    :return:
    """
    # check cache size
    if (
        not isinstance(cache_size, int) or
        cache_size <= 0
    ):
        raise RuntimeError("❌ 'simulation.general.cache_size' "
                           "must be an integer > 0.")


def _check_simulation_traditional_cache_params(fixed_ttl):
    """
    Method to check simulation traditional cache parameters.
    :param fixed_ttl: The fixed value of TTL.
    :return:
    """
    # check fixed ttl
    if (
        not isinstance(fixed_ttl, float) or
        fixed_ttl <= 0
    ):
        raise RuntimeError("❌ 'simulation.traditional_cache.ttl' "
                           "must be a float > 0.")


def _check_simulation_lstm_cache_params(
        prediction_interval,
        threshold_score,
        time_decay,
        ttl_base
):
    """
    Method to check simulation lstm cache parameters.
    :param prediction_interval: The prediction interval.
    :param threshold_score: The threshold for the score of keys.
    :param time_decay: The time decay to weight the score of keys.
    :param ttl_base: The TTL base value.
    :return:
    """
    # check prediction interval
    if (
        not isinstance(prediction_interval, int) or
        prediction_interval <= 0
    ):
        raise RuntimeError("❌ 'simulation.lstm_cache.prediction_interval' "
                           "must be a int > 0.")

    # check threshold score
    if not (
            isinstance(threshold_score, float)
            and 0.0 <= threshold_score <= 1.0
    ):
        raise RuntimeError(f"❌ 'simulation.lstm_cache.threshold_score' "
                           f"must be a float in [0.0, 1.0]")

    # check ttl base and time decay
    for name, val in [
        ("ttl_base", ttl_base),
        ("time_decay", time_decay),
    ]:
        if not (
            isinstance(val, float)
            and val > 0.0
        ):
            raise RuntimeError(f"❌ 'simulation.lstm_cache.{name}' "
                               f"must be a float > 0.")


def _validate_simulation_general_params(config):
    """
    Method to validate simulation general parameters.
    :param config: The config object.
    :return: The simulation general parameters.
    """
    # initial message
    info("🔄 Simulation general params validation started...")

    cache_size = get_config_value(
        config,
        "simulation.general.cache_size"
    )

    # check simulation general params
    _check_simulation_general_params(
        cache_size
    )

    # show a successful message
    info("🟢 Simulation general params validated.")

    return cache_size


def _validate_simulation_traditional_cache_params(config):
    """
    Method to validate simulation traditional cache parameters.
    :param config: The config object.
    :return: The simulation traditional cache parameters.
    """
    # initial message
    info("🔄 Simulation traditional cache params validation started...")

    fixed_ttl = get_config_value(
        config,
        "simulation.traditional_cache.ttl"
    )

    # check simulation traditional cache params
    _check_simulation_traditional_cache_params(
        fixed_ttl
    )

    # show a successful message
    info("🟢 Simulation traditional cache params validated.")

    return fixed_ttl


def _validate_simulation_lstm_cache_params(config):
    """
    Method to validate simulation lstm cache parameters.
    :param config: The config object.
    :return: The simulation lstm cache parameters.
    """
    # initial message
    info("🔄 Simulation lstm cache params validation started...")

    prediction_interval = get_config_value(
        config,
        "simulation.lstm_cache.prediction_interval"
    )
    threshold_score = get_config_value(
        config,
        "simulation.lstm_cache.threshold_score"
    )
    time_decay = get_config_value(
        config,
        "simulation.lstm_cache.time_decay"
    )
    ttl_base = get_config_value(
        config,
        "simulation.lstm_cache.ttl_base"
    )

    # check simulation lstm cache params
    _check_simulation_lstm_cache_params(
        prediction_interval,
        threshold_score,
        time_decay,
        ttl_base
    )

    # show a successful message
    info("🟢 Simulation lstm cache params validated.")

    return (
        prediction_interval,
        threshold_score,
        time_decay,
        ttl_base
    )