from config.config_io.config_reader import get_config
from config.validators.simulation_params.simulation_params_checker import check_simulation_general_params, \
    check_simulation_lstm_cache_params
from utils.logs.log_utils import info


def validate_simulation_general_params(config):
    """
    Method to validate simulation general parameters.
    :param config: The config object.
    :return: The simulation general parameters.
    """
    # initial message
    info("🔄 Simulation general params validation started...")

    cache_size = get_config(
        config,
        "simulation.general.cache_size"
    )
    ttl = get_config(
        config,
        "simulation.general.ttl"
    )

    # check simulation general params
    check_simulation_general_params(
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

    prediction_interval = get_config(
        config,
        "simulation.lstm_cache.prediction_interval"
    )
    threshold_score = get_config(
        config,
        "simulation.lstm_cache.threshold_score"
    )

    # check simulation lstm cache params
    check_simulation_lstm_cache_params(
        prediction_interval,
        threshold_score
    )

    # show a successful message
    info("🟢 Simulation lstm cache params validated.")

    return (
        prediction_interval,
        threshold_score
    )