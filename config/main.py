from config.utils.Config import Config
from config.validators.data_params_validator import \
    validate_data_sequence_params, validate_data_dataset_params, validate_data_access_temporal_pattern_params, \
    validate_data_access_behavior_pattern_params, validate_data_access_pattern_zipf_params, \
    validate_data_distribution_params
from config.validators.evaluation_params_validator import validate_evaluation_general_params
from config.validators.inference_params_validator import validate_inference_confidence_intervals_params
from config.validators.model_params_validator import validate_model_general_params, validate_model_params
from config.validators.simulation_params_validator import validate_simulation_general_params, \
    validate_simulation_lstm_cache_params
from config.validators.testing_params_validator import validate_testing_general_params
from config.validators.training_params_validator import validate_training_general_params, validate_training_optimizer_params, \
    validate_training_early_stopping_params
from config.validators.validation_params_validator import validate_cv_params, validate_validation_early_stopping_params, \
    validate_search_space_params
from config.config_io.config_loader import load_config
from utils.logs.log_utils import info, phase_var


def prepare_config():
    """
    Method to prepare the config file (loading, configuration, and validation).
    :return: A Config object containing all the configuration settings.
    """
    # initial message
    info("🔄 Config preparation started...")

    # set the variable indicating the state of the process
    phase_var.set("config_preparation")

    try:
        # load config file
        config_file = load_config()

        # data config
        (
            seed,
            distribution_type,
            num_requests,
            num_keys,
            first_key,
            last_key
        ) = validate_data_distribution_params(config_file)

        (
            zipf_alpha,
            zipf_alpha_start,
            zipf_alpha_end,
            zipf_time_steps,
        ) = validate_data_access_pattern_zipf_params(config_file)

        (
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
        ) = validate_data_access_behavior_pattern_params(config_file)

        (
            burst_high,
            burst_low,
            burst_hour_start,
            burst_hour_end,
            periodic_base_scale,
            periodic_amplitude
        ) = validate_data_access_temporal_pattern_params(config_file)

        (
            seq_len,
            embedding_dim,
            num_requests
        ) = validate_data_sequence_params(
            config_file,
            num_requests
        )

        (
            training_perc,
            validation_perc,
            static_save_path,
            dynamic_save_path
         ) = validate_data_dataset_params(config_file)

        # model config
        (
            num_features,
            model_save_path
        ) = validate_model_general_params(config_file)

        (
            model_params,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            proj_size
        ) = validate_model_params(config_file)

        # training config
        (
            training_num_epochs,
            training_batch_size
        ) = validate_training_general_params(config_file)

        (
            optimizer_type,
            learning_rate,
            weight_decay,
            momentum
        ) = validate_training_optimizer_params(config_file)

        (
            training_early_stopping_patience,
            training_early_stopping_delta
        ) = validate_training_early_stopping_params(config_file)

        # validation config
        (
            cv_num_folds,
            validation_num_epochs
        ) = validate_cv_params(config_file)

        (
            validation_early_stopping_patience,
            validation_early_stopping_delta
        ) = validate_validation_early_stopping_params(config_file)

        (
            search_space,
            hidden_size_range,
            num_layers_range,
            dropout_range,
            learning_rate_range
        ) = validate_search_space_params(config_file)

        # evaluation config
        top_k = validate_evaluation_general_params(config_file)

        # testing config
        testing_batch_size = validate_testing_general_params(config_file)

        # inference config
        (
            confidence_level,
            mc_dropout_num_samples
        ) = validate_inference_confidence_intervals_params(config_file)

        # simulation config
        (
            cache_size,
            ttl
        ) = validate_simulation_general_params(config_file)

        (
            prediction_interval,
            threshold_score
        ) = validate_simulation_lstm_cache_params(config_file)

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
            zipf_alpha=zipf_alpha,
            zipf_alpha_start=zipf_alpha_start,
            zipf_alpha_end=zipf_alpha_end,
            zipf_time_steps=zipf_time_steps,
            burst_high=burst_high,
            burst_low=burst_low,
            burst_hour_start=burst_hour_start,
            burst_hour_end=burst_hour_end,
            periodic_base_scale=periodic_base_scale,
            periodic_amplitude=periodic_amplitude,
            repetition_interval=repetition_interval,
            repetition_offset=repetition_offset,
            toggle_interval=toggle_interval,
            cycle_base=cycle_base,
            cycle_mod=cycle_mod,
            cycle_divisor=cycle_divisor,
            distortion_interval=distortion_interval,
            noise_range=noise_range,
            memory_interval=memory_interval,
            memory_offset=memory_offset,
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
            training_early_stopping_patience=training_early_stopping_patience,
            training_early_stopping_delta=training_early_stopping_delta,
            cv_num_folds=cv_num_folds,
            validation_num_epochs=validation_num_epochs,
            validation_early_stopping_patience=validation_early_stopping_patience,
            validation_early_stopping_delta=validation_early_stopping_delta,
            search_space=search_space,
            hidden_size_range=hidden_size_range,
            num_layers_range=num_layers_range,
            dropout_range=dropout_range,
            learning_rate_range=learning_rate_range,
            top_k=top_k,
            testing_batch_size=testing_batch_size,
            confidence_level=confidence_level,
            mc_dropout_num_samples=mc_dropout_num_samples,
            cache_size=cache_size,
            ttl=ttl,
            prediction_interval=prediction_interval,
            threshold_score=threshold_score,
        )

    except Exception as e:
        raise RuntimeError(f"❌ Error while preparing configuration: {e}.")
