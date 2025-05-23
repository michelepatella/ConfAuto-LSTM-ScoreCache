from config.Config import Config
from config.data_params_validator import _validate_data_distribution_params, _validate_data_access_pattern_params, \
    _validate_data_sequence_params, _validate_data_dataset_params
from config.evaluation_params_validator import _validate_evaluation_general_params
from config.inference_params_validator import _validate_inference_confidence_intervals_params
from config.model_params_validator import _validate_model_general_params, _validate_model_params
from config.testing_params_validator import _validate_testing_general_params
from config.training_params_validator import _validate_training_general_params, _validate_training_optimizer_params
from config.validation_params_validator import _validate_cv_params, _validate_early_stopping_params, \
    _validate_search_space_params
from config.config_utils import load_config
from utils.log_utils import info


def prepare_config():
    """
    Method to prepare the config file (loading, configuration, and validation).
    :return: A Config object containing all the configuration settings.
    """
    # initial message
    info("🔄 Config preparation started...")

    # load config file
    config_file = load_config()

    # data config
    (seed, distribution_type, num_requests,
     num_keys, first_key, last_key, freq_windows) = (
        _validate_data_distribution_params(config_file))

    (zipf_alpha, zipf_alpha_start, zipf_alpha_end,
     zipf_time_steps, locality_prob,
     burst_high, burst_low, burst_every, burst_peak,
     periodic_base_scale, periodic_amplitude) = (
        _validate_data_access_pattern_params(config_file))

    seq_len, embedding_dim, num_requests = (
        _validate_data_sequence_params(config_file, num_requests))

    training_perc, validation_perc, static_save_path, dynamic_save_path = (
        _validate_data_dataset_params(config_file))

    # model config
    num_features, model_save_path = (
        _validate_model_general_params(config_file))

    (model_params, hidden_size, num_layers,
     bias, batch_first, dropout, bidirectional, proj_size) = (
        _validate_model_params(config_file))

    # training config
    training_num_epochs, training_batch_size = (
        _validate_training_general_params(config_file))

    optimizer_type, learning_rate, weight_decay, momentum = (
        _validate_training_optimizer_params(config_file))

    # validation config
    cv_num_folds, validation_num_epochs = (
        _validate_cv_params(config_file))

    early_stopping_patience, early_stopping_delta = (
        _validate_early_stopping_params(config_file))

    (search_space, hidden_size_range, num_layers_range, dropout_range,
     learning_rate_range) = _validate_search_space_params(config_file)

    # evaluation config
    top_k = _validate_evaluation_general_params(config_file)

    # testing config
    testing_batch_size = _validate_testing_general_params(config_file)

    # inference config
    confidence_level = _validate_inference_confidence_intervals_params(config_file)

    # show a successful message
    info("✅ Config preparation completed.")

    return Config(
        config_file=config_file,
        seed=seed,
        distribution_type=distribution_type,
        num_requests=num_requests,
        num_keys=num_keys,
        first_key=first_key,
        last_key=last_key,
        freq_windows=freq_windows,
        zipf_alpha=zipf_alpha,
        zipf_alpha_start=zipf_alpha_start,
        zipf_alpha_end=zipf_alpha_end,
        zipf_time_steps=zipf_time_steps,
        locality_prob=locality_prob,
        burst_high=burst_high,
        burst_low=burst_low,
        burst_every=burst_every,
        burst_peak=burst_peak,
        periodic_base_scale=periodic_base_scale,
        periodic_amplitude=periodic_amplitude,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        training_perc=training_perc,
        validation_perc=validation_perc,
        static_save_path=static_save_path,
        dynamic_save_path=dynamic_save_path,
        num_features=num_features,
        model_save_path=model_save_path,
        model_params=model_params,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
        proj_size=proj_size,
        training_num_epochs=training_num_epochs,
        training_batch_size=training_batch_size,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        cv_num_folds=cv_num_folds,
        validation_num_epochs=validation_num_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_delta=early_stopping_delta,
        search_space=search_space,
        hidden_size_range=hidden_size_range,
        num_layers_range=num_layers_range,
        dropout_range=dropout_range,
        learning_rate_range=learning_rate_range,
        top_k=top_k,
        testing_batch_size=testing_batch_size,
        confidence_level=confidence_level
    )