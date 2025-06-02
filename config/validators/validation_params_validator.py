from utils.config_utils import get_config_value
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
        raise RuntimeError("❌ 'validation.cross_validation.num_folds' must be an integer > 1.")

    # check number of epochs
    if (
        validation_num_epochs <= 0 or
        not isinstance(validation_num_epochs, int)
    ):
        raise RuntimeError("❌ 'validation.cross_validation.num_epochs' must be an integer > 0.")


def _check_validation_early_stopping_params(
        validation_early_stopping_patience,
        validation_early_stopping_delta
):
    """
    Method to check validation early stopping parameters.
    :param validation_early_stopping_patience: Validation arly stopping patience value.
    :param validation_early_stopping_delta: Validation early stopping delta value.
    :return:
    """
    # check early stopping patience and delta
    for name, val, t, min_val in [
        ("validation.early_stopping.patience",
         validation_early_stopping_patience, int, 0),
        ("validation.early_stopping.delta",
         validation_early_stopping_delta, (int, float), 0)
    ]:
        if not (isinstance(val, t) and val >= min_val):
            raise RuntimeError(f"❌ '{name}' must be a "
                               f"{'number' if t == (int, float) else 'integer'} >= {min_val}.")


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
    # search space rules definition
    range_checks = [
        ("validation.search_space.model.params.hidden_size_range",
         hidden_size_range, int, lambda v: v > 0,
         "a non-empty list of integers > 0"),
        ("validation.search_space.model.params.num_layers_range",
         num_layers_range, int, lambda v: v > 0,
         "a non-empty list of integers > 0"),
        ("validation.search_space.model.params.dropout_range",
         dropout_range, float, lambda v: 0.0 <= v < 1.0,
         "a non-empty list of floats within [0.0, 1.0)"),
        ("validation.search_space.model.params.learning_rate_range",
         learning_rate_range, float, lambda v: v > 0,
         "a non-empty list of floats > 0"),
    ]

    for name, val, typ, cond, msg in range_checks:
        if (
            not isinstance(val, list)
            or len(val) == 0
            or not all(isinstance(v, typ) for v in val)
            or not all(cond(v) for v in val)
        ):
            raise RuntimeError(f"❌ '{name}' must be {msg}.")


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

    return (
        cv_num_folds,
        validation_num_epochs
    )


def _validate_validation_early_stopping_params(config):
    """
    Method to validate validation early stopping parameters.
    :param config: Config object.
    :return: All early stopping parameters.
    """
    # initial message
    info("🔄 Validation early stopping params validation started...")

    # early stopping
    validation_early_stopping_patience = get_config_value(
        config,
        "validation.early_stopping.patience"
    )
    validation_early_stopping_delta = get_config_value(
        config,
        "validation.early_stopping.delta"
    )

    # check early stopping params
    _check_validation_early_stopping_params(
        validation_early_stopping_patience,
        validation_early_stopping_delta
    )

    # show a successful message
    info("🟢 Validation early stopping params validated.")

    return (
        validation_early_stopping_patience,
        validation_early_stopping_delta
    )


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

    return (
        search_space,
        hidden_size_range,
        num_layers_range,
        dropout_range,
        learning_rate_range
    )