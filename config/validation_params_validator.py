from config.config_utils import get_config_value
from utils.log_utils import info


def _check_cv_params(
        cv_num_folds,
        validation_num_epochs
):
    """
    Method to check Cross-Validation parameters.
    :param cv_num_folds: The number of cross-validation folds.
    :param validation_num_epochs: The number of epochs used to validate the model.
    :return:
    """
    # check number of folds
    if (
        cv_num_folds <= 1 or
        not isinstance(cv_num_folds, int)
    ):
        raise RuntimeError("❌ 'validation.cross_validation.num_folds' "
                           "must be an integer > 1.")

    # check number of epochs
    if (
        validation_num_epochs <= 0 or
        not isinstance(validation_num_epochs, int)
    ):
        raise RuntimeError("❌ 'validation.cross_validation.num_epochs' "
                           "must be an integer > 0.")


def _check_early_stopping_params(
        early_stopping_patience,
        early_stopping_delta
):
    """
    Method to check early stopping parameters.
    :param early_stopping_patience: Early stopping patience value.
    :param early_stopping_delta: Early stopping delta value.
    :return:
    """
    # check patience
    if (
        not isinstance(early_stopping_patience, int)
        or early_stopping_patience < 0
    ):
        raise RuntimeError("❌ 'validation.early_stopping.patience' "
                           "must be an integer >= 0.")

    # check delta
    if (
        not isinstance(early_stopping_delta, (int, float)) or
        early_stopping_delta < 0
    ):
        raise RuntimeError("❌ 'validation.early_stopping.delta' "
                           "must be a number >= 0.")


def _check_search_space_params(
        hidden_size_range,
        num_layers_range,
        dropout_range,
        learning_rate_range
):
    """
    Method to check search space parameters.
    :param hidden_size_range: The hidden size range.
    :param num_layers_range: The number of layers range.
    :param dropout_range: The dropout range.
    :param learning_rate_range: The learning rate range.
    :return:
    """
    # check hidden size range
    if (
        not isinstance(hidden_size_range, list)
        or len(hidden_size_range) == 0 or
        not all(isinstance(v, int) for v in hidden_size_range) or
        not all(v > 0 for v in hidden_size_range)
    ):
        raise RuntimeError(f"❌ 'validation.search_space.model.params.hidden_size_range'"
                           f" must be a non-empty list of integers > 0.")

    # check number of layers range
    if (
        not isinstance(num_layers_range, list)
        or len(num_layers_range) == 0 or
        not all(isinstance(v, int) for v in num_layers_range) or
        not all(v > 0 for v in num_layers_range)
    ):
        raise RuntimeError(f"❌ 'validation.search_space.model.params.num_layers_range'"
                            f" must be a non-empty of integers > 0.")

    # check dropout range
    if (
        not isinstance(dropout_range, list)
        or len(dropout_range) == 0 or
        not all(isinstance(v, float) for v in dropout_range) or
        not all(0.0 <= v < 1.0 for v in dropout_range)
    ):
        raise RuntimeError(f"❌ 'validation.search_space.model.params.dropout_range'"
                            f" must be a non-empty list of floats within [0.0, 1.0).")

    # check learning rate range
    if (
        not isinstance(learning_rate_range, list)
        or len(learning_rate_range) == 0 or
        not all(isinstance(v, float) for v in learning_rate_range) or
        not all(v > 0 for v in learning_rate_range)
    ):
        raise RuntimeError(f"❌ 'validation.search_space.model.params.learning_rate_range'"
                           f" must be a non-empty list of floats > 0.")


def _validate_cv_params(config):
    """
    Method to validate cross validation parameters.
    :param config: Config object.
    :return: All the cross-validation parameters.
    """
    # initial message
    info("🔄 CV params validation started...")

    # cross-validation
    cv_num_folds = get_config_value(
        config,
        "validation.cross_validation.num_folds"
    )
    validation_num_epochs = get_config_value(
        config,
        "validation.cross_validation.num_epochs"
    )

    # check cross-validation params
    _check_cv_params(
        cv_num_folds,
        validation_num_epochs
    )

    # show a successful message
    info("🟢 CV params validated.")

    return cv_num_folds, validation_num_epochs


def _validate_early_stopping_params(config):
    """
    Method to validate early stopping parameters.
    :param config: Config object.
    :return: All early stopping parameters.
    """
    # initial message
    info("🔄 Early stopping params validation started...")

    # early stopping
    early_stopping_patience = get_config_value(
        config,
        "validation.early_stopping.patience"
    )
    early_stopping_delta = get_config_value(
        config,
        "validation.early_stopping.delta"
    )

    # check early stopping params
    _check_early_stopping_params(
        early_stopping_patience,
        early_stopping_delta
    )

    # show a successful message
    info("🟢 Early stopping params validated.")

    return early_stopping_patience, early_stopping_delta


def _validate_search_space_params(config):
    """
    Method to validate search space parameters.
    :param config: Config object.
    :return: All the search space parameters.
    """
    # initial message
    info("🔄 Search space params validation started...")

    # search space
    search_space = get_config_value(
        config,
        "validation.search_space"
    )
    hidden_size_range = get_config_value(
        config,
        "validation.search_space.model.params.hidden_size_range"
    )
    num_layers_range = get_config_value(
        config,
        "validation.search_space.model.params.num_layers_range"
    )
    dropout_range = get_config_value(
        config,
        "validation.search_space.model.params.dropout_range"
    )
    learning_rate_range = get_config_value(
        config,
        "validation.search_space.training.optimizer.learning_rate_range"
    )

    # check search space params
    _check_search_space_params(
        hidden_size_range,
        num_layers_range,
        dropout_range,
        learning_rate_range
    )

    # show a successful message
    info("🟢 Search space params validated.")

    return (search_space, hidden_size_range, num_layers_range,
            dropout_range, learning_rate_range)