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
        cv_num_folds <= 0 or
        not isinstance(cv_num_folds, int)
    ):
        raise RuntimeError("❌ 'validation.cross_validation.num_folds' "
                           "must be an integer > 0.")

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
        or early_stopping_patience <=0
    ):
        raise RuntimeError("❌ 'validation.early_stopping.patience' "
                           "must be an integer > 0.")

    # check delta
    if (
        not isinstance(early_stopping_delta, (int, float)) or
        early_stopping_delta < 0
    ):
        raise RuntimeError("❌ 'validation.early_stopping.delta' "
                           "must be a number >= 0.")