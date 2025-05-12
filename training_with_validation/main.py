import logging
from torch import nn
from utils.AccessLogsDataset import AccessLogsDataset
from utils.config_utils import load_config, get_config_value
from training_with_validation.best_params_saver import _save_best_params
from training_with_validation.grid_search_optimizer import _grid_search


def training_with_validation():
    """
    Method to orchestrate training and validation of the model.
    :return:
    """
    # load config file
    config = load_config()

    # load the training set
    training_set = AccessLogsDataset(
        get_config_value(config, "data.static_dataset_path"),
        "training"
    )

    # grid search for best parameters
    best_params = _grid_search(training_set)

    # set the best parameters
    _save_best_params(best_params)

    logging.info("Training with validation successfully completed.")