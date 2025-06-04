from config.config_io.config_reader import get_config
from config.validators.data_params.data_params_checker import check_distribution_params, check_zipf_params, \
    check_access_behavior_params, check_temporal_pattern_params, check_sequence_params, check_dataset_params
from utils.logs.log_utils import info


def validate_data_distribution_params(config):
    """
    Method to validate data distribution parameters.
    :param config: The configuration object.
    :return: All the data distribution parameters.
    """
    # initial message
    info("🔄 Data distribution params validation started...")

    # distribution
    seed = get_config(
        config,
        "data.distribution.seed"
    )
    distribution_type = get_config(
        config,
        "data.distribution.type"
    )
    num_requests = get_config(
        config,
        "data.distribution.num_requests"
    )
    num_keys = get_config(
        config,
        "data.distribution.num_keys"
    )
    first_key = get_config(
        config,
        "data.distribution.key_range.first_key"
    )
    last_key = get_config(
        config,
        "data.distribution.key_range.last_key"
    ) + 1

    # check distribution params
    check_distribution_params(
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
    zipf_alpha = get_config(
        config,
        "data.access_pattern.zipf.alpha"
    )
    zipf_alpha_start = get_config(
        config,
        "data.access_pattern.zipf.alpha_start"
    )
    zipf_alpha_end = get_config(
        config,
        "data.access_pattern.zipf.alpha_end"
    )
    zipf_time_steps = get_config(
        config,
        "data.access_pattern.zipf.time_steps"
    )

    # check zipf parameters
    check_zipf_params(
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
    repetition_interval = get_config(
        config,
        "data.access_pattern.access_behavior.repetition_interval"
    )
    repetition_offset = get_config(
        config,
        "data.access_pattern.access_behavior.repetition_offset"
    )
    toggle_interval = get_config(
        config,
        "data.access_pattern.access_behavior.toggle_interval"
    )
    cycle_base = get_config(
        config,
        "data.access_pattern.access_behavior.cycle_base"
    )
    cycle_mod = get_config(
        config,
        "data.access_pattern.access_behavior.cycle_mod"
    )
    cycle_divisor = get_config(
        config,
        "data.access_pattern.access_behavior.cycle_divisor"
    )
    distortion_interval = get_config(
        config,
        "data.access_pattern.access_behavior.distortion_interval"
    )
    noise_range = get_config(
        config,
        "data.access_pattern.access_behavior.noise_range"
    )
    memory_interval = get_config(
        config,
        "data.access_pattern.access_behavior.memory_interval"
    )
    memory_offset = get_config(
        config,
        "data.access_pattern.access_behavior.memory_offset"
    )

    # check access behavior pattern parameters
    check_access_behavior_params(
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
    burst_high = get_config(
        config,
        "data.temporal_pattern.burstiness.burst_high"
    )
    burst_low = get_config(
        config,
        "data.temporal_pattern.burstiness.burst_low"
    )
    burst_hour_start = get_config(
        config,
        "data.temporal_pattern.burstiness.burst_hour_start"
    )
    burst_hour_end = get_config(
        config,
        "data.temporal_pattern.burstiness.burst_hour_end"
    )
    # periodic
    periodic_base_scale = get_config(
        config,
        "data.temporal_pattern.periodic.base_scale"
    )
    periodic_amplitude = get_config(
        config,
        "data.temporal_pattern.periodic.amplitude"
    )

    # check temporal pattern params
    check_temporal_pattern_params(
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
    seq_len = get_config(
        config,
        "data.sequence.len"
    )
    embedding_dim = get_config(
        config,
        "data.sequence.embedding_dim"
    )

    # check sequence params
    check_sequence_params(
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
    training_perc = get_config(
        config,
        "data.dataset.training_perc"
    )
    validation_perc = get_config(
        config,
        "data.dataset.validation_perc"
    )
    static_save_path = get_config(
        config,
        "data.dataset.static_save_path"
    )
    dynamic_save_path = get_config(
        config,
        "data.dataset.dynamic_save_path"
    )

    # check dataset params
    check_dataset_params(
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