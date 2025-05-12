import logging
from utils.AccessLogsDataset import AccessLogsDataset
from utils.config_utils import _get_config_value
from training_with_validation.best_params_saver import _save_best_params
from training_with_validation.grid_search_optimizer import _grid_search


def training_with_validation():
    """
    Method to orchestrate training and validation of the model.
    :return:
    """
    # load the training set
    training_set = AccessLogsDataset(
        _get_config_value("data.static_dataset_path"),
        "training"
    )

    # grid search for best parameters
    best_params = _grid_search(training_set)

    # set the best parameters
    _save_best_params(best_params)

    logging.info("Training with validation successfully completed.")