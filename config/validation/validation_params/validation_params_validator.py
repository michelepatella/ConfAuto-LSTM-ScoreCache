from config.config_io.config_reader import get_config
from config.validation.validation_params.validation_params_checker import check_cv_params, \
    check_validation_early_stopping_params, check_search_space_params
from utils.logs.log_utils import info


def validate_cv_params(config):
    """
    Method to validate cross validation parameters.
    :param config: Config object.
    :return: All the cross-validation parameters.
    """
    # initial message
    info("🔄 CV params validation started...")

    # cross-validation
    cv_num_folds = get_config(
        config,
        "validation.cross_validation.num_folds"
    )
    validation_num_epochs = get_config(
        config,
        "validation.cross_validation.num_epochs"
    )

    # check cross-validation params
    check_cv_params(
        cv_num_folds,
        validation_num_epochs
    )

    # show a successful message
    info("🟢 CV params validated.")

    return (
        cv_num_folds,
        validation_num_epochs
    )


def validate_validation_early_stopping_params(config):
    """
    Method to validate validation early stopping parameters.
    :param config: Config object.
    :return: All early stopping parameters.
    """
    # initial message
    info("🔄 Validation early stopping params validation started...")

    # early stopping
    validation_early_stopping_patience = get_config(
        config,
        "validation.early_stopping.patience"
    )
    validation_early_stopping_delta = get_config(
        config,
        "validation.early_stopping.delta"
    )

    # check early stopping params
    check_validation_early_stopping_params(
        validation_early_stopping_patience,
        validation_early_stopping_delta
    )

    # show a successful message
    info("🟢 Validation early stopping params validated.")

    return (
        validation_early_stopping_patience,
        validation_early_stopping_delta
    )


def validate_search_space_params(config):
    """
    Method to validate search space parameters.
    :param config: Config object.
    :return: All the search space parameters.
    """
    # initial message
    info("🔄 Search space params validation started...")

    # search space
    search_space = get_config(
        config,
        "validation.search_space"
    )
    hidden_size_range = get_config(
        config,
        "validation.search_space.model.params.hidden_size_range"
    )
    num_layers_range = get_config(
        config,
        "validation.search_space.model.params.num_layers_range"
    )
    dropout_range = get_config(
        config,
        "validation.search_space.model.params.dropout_range"
    )
    learning_rate_range = get_config(
        config,
        "validation.search_space.training.optimizer.learning_rate_range"
    )

    # check search space params
    check_search_space_params(
        hidden_size_range,
        num_layers_range,
        dropout_range,
        learning_rate_range
    )

    # show a successful message
    info("🟢 Search space params validated.")

    return (
        search_space,
        hidden_size_range,
        num_layers_range,
        dropout_range,
        learning_rate_range
    )