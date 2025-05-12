import logging
from tqdm import tqdm
from utils.config_utils import _load_config, _get_config_value
from training_with_validation.best_params_updater import _check_and_update_best_params
from training_with_validation.time_series_cv import _time_series_cv

def _parameter_combination():
    """
    Method to combine the parameters of each fold iteration.
    """
    config = _load_config()

    # define the parameters combination
    param_combinations = [
        (hidden_size, num_layers, dropout, learning_rate)
        for hidden_size in _get_config_value(
            config,
            "validation.hidden_size_range"
        )
        for num_layers in _get_config_value(
            config,
            "validation.num_layers_range"
        )
        for dropout in _get_config_value(
            config,
            "validation.dropout_range"
        )
        for learning_rate in _get_config_value(
            config,
            "validation.learning_rate_range"
        )
    ]

    # check the parameters combination calculated
    if not param_combinations:
        raise Exception("No parameter combinations found while performing Grid Search.")

    return param_combinations

def _grid_search(training_set):
    """
    Method to perform grid search to find the best parameters.
    :param training_set: The training set.
    :return: The best parameters.
    """
    # initialize the best parameters and average loss
    best_params = {}
    best_avg_loss = float("inf")

    # get the parameters combination
    param_combinations = _parameter_combination()

    # grid search
    with tqdm(
            total=len(param_combinations),
            desc="Grid Search Progress"
    ) as pbar:
        for hidden_size, num_layers, dropout, learning_rate in param_combinations:

            fold_losses = []

            # perform the time series CV
            fold_losses = _time_series_cv(
                training_set,
                hidden_size,
                num_layers,
                dropout,
                learning_rate,
                fold_losses
            )

            # group current parameters together
            curr_params = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": learning_rate
            }

            # check the loss and eventually update the best parameters
            best_avg_loss, best_params = _check_and_update_best_params(
                fold_losses,
                best_avg_loss,
                curr_params,
                best_params
            )

            # update the progress bar
            pbar.update(1)

    # print the best params found
    logging.info(f"Best params found: {best_params}")

    # print the best average loss found
    logging.info(f"Best avg loss: {best_avg_loss}")

    return best_params