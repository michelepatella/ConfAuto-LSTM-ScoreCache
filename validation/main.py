import logging
from utils.setup_utils import _loader_setup
from validation.best_params_saver import _save_best_params
from validation.grid_search_optimizer import _grid_search
from utils.dataset_utils import _get_dataset_path_type


def validation():
    """
    Method to orchestrate the validation of the model.
    :return:
    """
    # initial message
    logging.info("🔄 Validation started...")

    # load the training set
    training_set = _loader_setup("training")

    # grid search for best parameters
    best_params = _grid_search(training_set)

    # set the best parameters
    _save_best_params(best_params)

    # print a successful message
    logging.info("✅ Validation successfully completed.")