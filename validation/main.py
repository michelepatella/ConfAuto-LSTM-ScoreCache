from utils.log_utils import _info, phase_var
from utils.data_utils import _loader_setup
from validation.best_params_saver import _save_best_params
from validation.grid_search_optimizer import _grid_search


def validation():
    """
    Method to orchestrate the validation of the model.
    :return:
    """
    # initial message
    _info("🔄 Validation started...")

    # set the variable indicating the state of the process
    phase_var.set("validation")

    # load the training set
    training_set, _ = _loader_setup(
        "training",
        False
    )

    # grid search for best parameters
    best_params = _grid_search(training_set)

    # set the best parameters
    _save_best_params(best_params)

    # print a successful message
    _info("✅ Validation successfully completed.")