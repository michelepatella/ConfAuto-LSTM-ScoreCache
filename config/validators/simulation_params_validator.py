from utils.config_utils import get_config_value
from utils.log_utils import info


def _check_simulation_general_params(
        cache_size,
        ttl
):
    """
    Method to check simulation general parameters.
    :param cache_size: The size of the cache.
    :param ttl: The TTL of the cache.
    :return:
    """
    # check cache size
    if (
        not isinstance(cache_size, int) or
        cache_size <= 0
    ):
        raise RuntimeError("❌ 'simulation.general.cache_size' must be an integer > 0.")

    # check fixed ttl
    if (
        not isinstance(ttl, int) or
        ttl <= 0
    ):
        raise RuntimeError("❌ 'simulation.general.ttl' must be a int > 0.")


def _check_simulation_lstm_cache_params(
        prediction_interval,
        threshold_score
):
    """
    Method to check simulation lstm cache parameters.
    :param prediction_interval: The prediction interval.
    :param threshold_score: The threshold for the score of keys.
    :return:
    """
    # check prediction interval
    if (
        not isinstance(prediction_interval, int) or
        prediction_interval <= 0
    ):
        raise RuntimeError("❌ 'simulation.lstm_cache.prediction_interval' must be a int > 0.")

    # check threshold score
    if not (
            isinstance(threshold_score, float)
            and 0.0 <= threshold_score <= 1.0
    ):
        raise RuntimeError(f"❌ 'simulation.lstm_cache.threshold_score' must be a float in [0.0, 1.0]")


def validate_simulation_general_params(config):
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
    ttl = get_config_value(
        config,
        "simulation.general.ttl"
    )

    # check simulation general params
    _check_simulation_general_params(
        cache_size,
        ttl
    )

    # show a successful message
    info("🟢 Simulation general params validated.")

    return cache_size, ttl


def validate_simulation_lstm_cache_params(config):
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

    # check simulation lstm cache params
    _check_simulation_lstm_cache_params(
        prediction_interval,
        threshold_score
    )

    # show a successful message
    info("🟢 Simulation lstm cache params validated.")

    return (
        prediction_interval,
        threshold_score
    )