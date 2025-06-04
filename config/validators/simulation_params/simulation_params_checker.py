def check_simulation_general_params(
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


def check_simulation_lstm_cache_params(
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