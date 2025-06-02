from utils.config_utils import get_config_value
from utils.log_utils import info


def _check_distribution_params(
        seed,
        distribution_type,
        num_requests,
        num_keys,
        first_key,
        last_key
):
    """
    Method to check data distribution parameters.
    :param seed: The random seed.
    :param distribution_type: The distribution type.
    :param num_requests: The number of requests.
    :param num_keys: The number of keys.
    :param first_key: The first key.
    :param last_key: The last key.
    :return:
    """
    # check seed
    if not (
            isinstance(seed, int)
            and seed >= 0
    ):
        raise RuntimeError("❌ 'data.distribution.seed' must be an integer >= 0.")

    # check distribution type
    if distribution_type not in {"static", "dynamic"}:
        raise RuntimeError("❌ 'data.distribution.type' must be 'static' or 'dynamic'.")

    # check integer params with limits
    for name, val, min_val in [
        ("num_requests", num_requests, 1),
        ("num_keys", num_keys, 2),
    ]:
        if not (
                isinstance(val, int)
                and val >= min_val
        ):
            raise RuntimeError(f"❌ 'data.distribution.{name}' must be an integer >= {min_val}.")

    # check first_key and last_key are ints and last_key > first_key
    if not (
            isinstance(first_key, int)
            and isinstance(last_key, int)
            and last_key > first_key
    ):
        raise RuntimeError("❌ 'data.distribution.key_range.first_key' and 'data.distribution.key_range.last_key' "
            "must be integers with first_key < last_key."
        )

    # check num_keys matches the key range length
    if abs(last_key - first_key) != num_keys:
        raise RuntimeError("❌ 'data.distribution.num_keys' must match the range defined by first_key and last_key.")


def _check_zipf_params(
        zipf_alpha,
        zipf_alpha_start,
        zipf_alpha_end,
        zipf_time_steps
):
    """
    Method to check Zipf parameters.
    :param zipf_alpha: The Zipf parameter.
    :param zipf_alpha_start: The initial Zipf parameter.
    :param zipf_alpha_end: The final Zipf parameter.
    :param zipf_time_steps: The time steps.
    :return:
    """
    # check alpha parameter
    if (
        not isinstance(zipf_alpha, float)
        or zipf_alpha <= 0
    ):
        raise RuntimeError("❌ 'data.access_pattern.zipf.alpha' must be a float > 0.")

    # check alpha start and alpha end
    if not all(
            isinstance(v, float) and v > 0
            for v in [zipf_alpha_start, zipf_alpha_end]
    ):
        raise RuntimeError(
            "❌ 'data.access_pattern.zipf.alpha_start' and "
            "'data.access_pattern.zipf.alpha_end' must be float > 0."
        )

    if zipf_alpha_end < zipf_alpha_start:
        raise RuntimeError(
            "❌ 'data.access_pattern.zipf.alpha_end' "
            "must be >= 'data.access_pattern.zipf.alpha_start'."
        )

    # check time steps
    if not (
            isinstance(zipf_time_steps, int)
            and zipf_time_steps > 0
    ):
        raise RuntimeError("❌ 'data.access_pattern.zipf.time_steps' must be an integer > 0.")


def _check_access_behavior_params(
    repetition_interval,
    repetition_offset,
    toggle_interval,
    cycle_base,
    cycle_mod,
    cycle_divisor,
    distortion_interval,
    noise_range,
    memory_interval,
    memory_offset
):
    # check some values (that must be int >= 0)
    for name, val in [
        ("repetition_interval", repetition_interval),
        ("repetition_offset", repetition_offset),
        ("toggle_interval", toggle_interval),
        ("distortion_interval", distortion_interval),
        ("memory_interval", memory_interval),
        ("memory_offset", memory_offset)
    ]:
        if not isinstance(val, int) or val < 0:
            raise RuntimeError(f"❌ 'data.access_pattern.access_behavior.{name}' must be an integer >= 0.")

    # check cycle parameters
    for name, val in [
        ("cycle_base", cycle_base),
        ("cycle_mod", cycle_mod),
        ("cycle_divisor", cycle_divisor)
    ]:
        if not isinstance(val, int) or val <= 0:
            raise RuntimeError(f"❌ 'data.access_pattern.access_behavior.{name}' must be an integer > 0.")

    # check noise range
    if (
        not isinstance(noise_range, (list, tuple))
        or len(noise_range) != 2
        or not all(isinstance(v, (int, float)) for v in noise_range)
    ):
        raise RuntimeError(
            "❌ 'data.access_pattern.access_behavior.noise_range' "
            "must be a list or tuple of two numbers.")


def _check_temporal_pattern_params(
    burst_high,
    burst_low,
    burst_hour_start,
    burst_hour_end,
    periodic_base_scale,
    periodic_amplitude
):
    """
    Method to check temporal pattern parameters.
    :param burst_high: The burst high value.
    :param burst_low: The burst low value.
    :param burst_hour_start: The burst hour start value.
    :param burst_hour_end: The burst hour end value.
    :param periodic_base_scale: The periodic base scale.
    :param periodic_amplitude: The periodic amplitude.
    :return:
    """
    # check burst high and low
    for name, val in [
        ("burst_high", burst_high),
        ("burst_low", burst_low)
    ]:
        if not (
                isinstance(val, float)
                and val > 0
        ):
            raise RuntimeError(f"❌ 'data.temporal_pattern.burstiness.{name}' must be a float > 0.")

    if burst_high >= burst_low:
        raise RuntimeError(
            "❌ 'data.temporal_pattern.burstiness.burst_high' "
            "must be < 'data.temporal_pattern.burstiness.burst_low'."
        )

    # check burst start
    if not (
            isinstance(burst_hour_start, int)
            and 0 <= burst_hour_start <= 23
    ):
        raise RuntimeError("❌ 'data.temporal_pattern.burstiness.burst_hour_start' must be an integer in [0, 23].")

    # check burst end
    if not (
            isinstance(burst_hour_end, int)
            and 0 <= burst_hour_end <= 23
    ):
        raise RuntimeError("❌ 'data.temporal_pattern.burstiness.burst_hour_end' must be an integer in [0, 23].")

    # check periodic base scale
    if not (
            isinstance(periodic_base_scale, (int, float))
            and periodic_base_scale > 0
    ):
        raise RuntimeError("❌ 'data.temporal_pattern.periodic.base_scale' must be a number > 0.")

    # check periodic amplitude
    if not (
            isinstance(periodic_amplitude, (int, float))
            and periodic_amplitude >= 0
    ):
        raise RuntimeError("❌ 'data.temporal_pattern.periodic.periodic_amplitude' must be a number >= 0.")


def _check_sequence_params(
        seq_len,
        embedding_dim,
        num_requests
):
    """
    Method to check sequence parameters.
    :param seq_len: The sequence length.
    :param embedding_dim: The embedding dimension.
    :param num_requests: The number of requests.
    :return:
    """
    # check sequence length and embedding dimension are integers > 0
    for name, val in [
        ("len", seq_len),
        ("embedding_dim", embedding_dim)
    ]:
        if not (
                isinstance(val, int)
                and val > 0
        ):
            raise RuntimeError(f"❌ 'data.sequence.{name}' must be an integer > 0.")

    # check seq_len <= num_requests
    if seq_len > num_requests:
        raise RuntimeError("❌ 'data.sequence.len' must be <= 'data.distribution.num_requests'.")


def _check_dataset_params(
        training_perc,
        validation_perc,
        static_save_path,
        dynamic_save_path
):
    """
    Method to check dataset parameters.
    :param training_perc: The training set percentage.
    :param validation_perc: The validation set percentage.
    :param static_save_path: The path to save the static dataset.
    :param dynamic_save_path: The path to save the dynamic dataset.
    :return:
    """
    # check training percentage
    if not (
            isinstance(training_perc, float)
            and 0.0 <= training_perc <= 1.0
    ):
        raise RuntimeError("❌ 'data.dataset.training_perc' must be a float between 0.0 and 1.0.")

    # check validation percentage
    if not (
            isinstance(validation_perc, float)
            and 0.0 <= validation_perc < 1.0
    ):
        raise RuntimeError("❌ 'data.dataset.validation_perc' must be a float between 0.0 and 1.0 (excluded).")

    # check dataset paths
    if not (
            isinstance(static_save_path, str)
            and isinstance(dynamic_save_path, str)
    ):
        raise RuntimeError(
            "❌ Both 'data.dataset.static_save_path' and "
            "'data.dataset.dynamic_save_path' must be strings."
        )


def validate_data_distribution_params(config):
    """
    Method to validate data distribution parameters.
    :param config: The configuration object.
    :return: All the data distribution parameters.
    """
    # initial message
    info("🔄 Data distribution params validation started...")

    # distribution
    seed = get_config_value(
        config,
        "data.distribution.seed"
    )
    distribution_type = get_config_value(
        config,
        "data.distribution.type"
    )
    num_requests = get_config_value(
        config,
        "data.distribution.num_requests"
    )
    num_keys = get_config_value(
        config,
        "data.distribution.num_keys"
    )
    first_key = get_config_value(
        config,
        "data.distribution.key_range.first_key"
    )
    last_key = get_config_value(
        config,
        "data.distribution.key_range.last_key"
    ) + 1

    # check distribution params
    _check_distribution_params(
        seed,
        distribution_type,
        num_requests,
        num_keys,
        first_key,
        last_key
    )

    # show a successful message
    info("🟢 Data distribution params validated.")

    return (
        seed,
        distribution_type,
        num_requests,
        num_keys,
        first_key,
        last_key
    )


def validate_data_access_pattern_zipf_params(config):
    """
    Method to validate data access pattern Zipf parameters.
    :param config: The configuration object.
    :return: All the data access pattern Zipf parameters.
    """
    # initial message
    info("🔄 Data access pattern Zipf params validation started...")

    # access pattern
    # zipf
    zipf_alpha = get_config_value(
        config,
        "data.access_pattern.zipf.alpha"
    )
    zipf_alpha_start = get_config_value(
        config,
        "data.access_pattern.zipf.alpha_start"
    )
    zipf_alpha_end = get_config_value(
        config,
        "data.access_pattern.zipf.alpha_end"
    )
    zipf_time_steps = get_config_value(
        config,
        "data.access_pattern.zipf.time_steps"
    )

    # check zipf parameters
    _check_zipf_params(
        zipf_alpha,
        zipf_alpha_start,
        zipf_alpha_end,
        zipf_time_steps
    )

    # show a successful message
    info("🟢 Data access pattern Zipf params validated.")

    return (
        zipf_alpha,
        zipf_alpha_start,
        zipf_alpha_end,
        zipf_time_steps,
    )


def validate_data_access_behavior_pattern_params(config):
    """
    Method to validate data access behavior pattern parameters.
    :param config: The configuration object.
    :return: All the data access behavior pattern parameters.
    """
    # initial message
    info("🔄 Data access behavior pattern params validation started...")

    # access behavior
    repetition_interval = get_config_value(
        config,
        "data.access_pattern.access_behavior.repetition_interval"
    )
    repetition_offset = get_config_value(
        config,
        "data.access_pattern.access_behavior.repetition_offset"
    )
    toggle_interval = get_config_value(
        config,
        "data.access_pattern.access_behavior.toggle_interval"
    )
    cycle_base = get_config_value(
        config,
        "data.access_pattern.access_behavior.cycle_base"
    )
    cycle_mod = get_config_value(
        config,
        "data.access_pattern.access_behavior.cycle_mod"
    )
    cycle_divisor = get_config_value(
        config,
        "data.access_pattern.access_behavior.cycle_divisor"
    )
    distortion_interval = get_config_value(
        config,
        "data.access_pattern.access_behavior.distortion_interval"
    )
    noise_range = get_config_value(
        config,
        "data.access_pattern.access_behavior.noise_range"
    )
    memory_interval = get_config_value(
        config,
        "data.access_pattern.access_behavior.memory_interval"
    )
    memory_offset = get_config_value(
        config,
        "data.access_pattern.access_behavior.memory_offset"
    )

    # check access behavior pattern parameters
    _check_access_behavior_params(
        repetition_interval,
        repetition_offset,
        toggle_interval,
        cycle_base,
        cycle_mod,
        cycle_divisor,
        distortion_interval,
        noise_range,
        memory_interval,
        memory_offset
    )

    # show a successful message
    info("🟢 Data access behavior pattern params validated.")

    return (
        repetition_interval,
        repetition_offset,
        toggle_interval,
        cycle_base,
        cycle_mod,
        cycle_divisor,
        distortion_interval,
        noise_range,
        memory_interval,
        memory_offset,
    )


def validate_data_access_temporal_pattern_params(config):
    """
    Method to validate data access temporal pattern parameters.
    :param config: The configuration object.
    :return: All the data access temporal pattern parameters.
    """
    # initial message
    info("🔄 Data access temporal pattern params validation started...")

    # temporal pattern
    # burstiness
    burst_high = get_config_value(
        config,
        "data.temporal_pattern.burstiness.burst_high"
    )
    burst_low = get_config_value(
        config,
        "data.temporal_pattern.burstiness.burst_low"
    )
    burst_hour_start = get_config_value(
        config,
        "data.temporal_pattern.burstiness.burst_hour_start"
    )
    burst_hour_end = get_config_value(
        config,
        "data.temporal_pattern.burstiness.burst_hour_end"
    )
    # periodic
    periodic_base_scale = get_config_value(
        config,
        "data.temporal_pattern.periodic.base_scale"
    )
    periodic_amplitude = get_config_value(
        config,
        "data.temporal_pattern.periodic.amplitude"
    )

    # check temporal pattern params
    _check_temporal_pattern_params(
        burst_high,
        burst_low,
        burst_hour_start,
        burst_hour_end,
        periodic_base_scale,
        periodic_amplitude
    )

    # show a successful message
    info("🟢 Data access temporal pattern params validated.")

    return (
        burst_high,
        burst_low,
        burst_hour_start,
        burst_hour_end,
        periodic_base_scale,
        periodic_amplitude
    )


def validate_data_sequence_params(config, num_requests):
    """
    Method to validate data sequence parameters.
    :param config: The configuration object.
    :param num_requests: The number of requests.
    :return: All the data sequence parameters.
    """
    # initial message
    info("🔄 Data sequence params validation started...")

    # sequence
    seq_len = get_config_value(
        config,
        "data.sequence.len"
    )
    embedding_dim = get_config_value(
        config,
        "data.sequence.embedding_dim"
    )

    # check sequence params
    _check_sequence_params(
        seq_len,
        embedding_dim,
        num_requests
    )

    # show a successful message
    info("🟢 Data sequence params validated.")

    return (
        seq_len,
        embedding_dim,
        num_requests
    )


def validate_data_dataset_params(config):
    """
    Method to validate data dataset parameters.
    :param config: The configuration object.
    :return: All the data dataset parameters.
    """
    # initial message
    info("🔄 Dataset params validation started...")

    # dataset
    training_perc = get_config_value(
        config,
        "data.dataset.training_perc"
    )
    validation_perc = get_config_value(
        config,
        "data.dataset.validation_perc"
    )
    static_save_path = get_config_value(
        config,
        "data.dataset.static_save_path"
    )
    dynamic_save_path = get_config_value(
        config,
        "data.dataset.dynamic_save_path"
    )

    # check dataset params
    _check_dataset_params(
        training_perc,
        validation_perc,
        static_save_path,
        dynamic_save_path
    )

    # show a successful message
    info("🟢 Dataset params validated.")

    return (
        training_perc,
        validation_perc,
        static_save_path,
        dynamic_save_path
    )