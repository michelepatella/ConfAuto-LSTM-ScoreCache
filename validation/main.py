import logging
from torch import nn
from utils.access_log_dataset import AccessLogsDataset
from utils.config_loader import load_config
from validation.best_params_saver import _save_best_params
from validation.grid_search_optimizer import _grid_search


def parameter_tuning():
    """
    Method to orchestrate the parameter tuning of the model.
    :return:
    """
    # load config file
    config = load_config()

    # load data config
    if config is not None and "data" in config:
        data_config = config["data"]
    else:
        return

    # try to load the dataset
    try:
        # load the dataset
        dataset = AccessLogsDataset(data_config["static_dataset_path"], "validation")
    except FileNotFoundError as e:
        logging.error(f"Error opening the file: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error while loading dataset: {e}")
        return

    # define the loss function
    criterion = nn.CrossEntropyLoss()

    # grid search for best parameters
    best_params = _grid_search(dataset, criterion)

    # set the best parameters
    _save_best_params(best_params)

if __name__ == "__main__":
    parameter_tuning()