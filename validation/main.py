import logging
from torch import nn
from utils.AccessLogsDataset import AccessLogsDataset
from utils.config_utils import load_config, get_config_value
from validation.best_params_saver import _save_best_params
from validation.grid_search_optimizer import _grid_search


def validation():
    """
    Method to orchestrate the validation of the model, to find
    the best parameters.
    :return:
    """
    # load config file
    config = load_config()

    # try to load the dataset
    try:
        # load the dataset
        dataset = AccessLogsDataset(
            get_config_value(config,"data.static_dataset_path"),
            "validation"
        )
    except Exception as e:
        raise Exception(f"An unexpected error while loading dataset: {e}")

    # define the loss function
    criterion = nn.CrossEntropyLoss()

    try:
        # grid search for best parameters
        best_params = _grid_search(dataset, criterion)

        # set the best parameters
        _save_best_params(best_params)
    except Exception as e:
        raise Exception(f"Validation failed: {e}")

    logging.info(f"Validation successfully completed.")