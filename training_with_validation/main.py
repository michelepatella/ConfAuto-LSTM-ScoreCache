import logging
from utils.AccessLogsDataset import AccessLogsDataset
from training_with_validation.best_params_saver import _save_best_params
from training_with_validation.grid_search_optimizer import _grid_search
from utils.config_utils import _get_config_value
from utils.dataset_utils import _get_dataset_path_type


def training_with_validation():
    """
    Method to orchestrate training with validation of the model.
    :return:
    """
    # initial message
    logging.info("🔄 Training with validation started...")

    # get the dataset path
    dataset_path,_ = _get_dataset_path_type()

    # load the training set
    training_set = AccessLogsDataset(
        _get_config_value(dataset_path),
        "training"
    )

    # grid search for best parameters
    best_params = _grid_search(training_set)

    # set the best parameters
    _save_best_params(best_params)

    # print a successful message
    logging.info("✅ Training with validation successfully completed.")