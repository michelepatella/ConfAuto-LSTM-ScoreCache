def check_cv_params(
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
        raise RuntimeError("'validation.cross_validation.num_folds' must be an integer > 1.")

    # check number of epochs
    if (
        validation_num_epochs <= 0 or
        not isinstance(validation_num_epochs, int)
    ):
        raise RuntimeError("'validation.cross_validation.num_epochs' must be an integer > 0.")


def check_validation_early_stopping_params(
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
            raise RuntimeError(f"'{name}' must be a "
                               f"{'number' if t == (int, float) else 'integer'} >= {min_val}.")


def check_search_space_params(
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
            raise RuntimeError(f"'{name}' must be {msg}.")