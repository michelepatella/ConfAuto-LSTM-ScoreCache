import logging
from utils.AccessLogsDataset import AccessLogsDataset
from validation.best_params_saver import _save_best_params
from validation.grid_search_optimizer import _grid_search
from utils.config_utils import _get_config_value
from utils.dataset_utils import _get_dataset_path_type


def validation():
    """
    Method to orchestrate the validation of the model.
    :return:
    """
    # initial message
    logging.info("🔄 Validation started...")

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
    logging.info("✅ Validation successfully completed.")