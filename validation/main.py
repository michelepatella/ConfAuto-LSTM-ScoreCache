from utils.data.AccessLogsDataset import AccessLogsDataset
from utils.logs.log_utils import info, phase_var
from utils.data.dataloader.dataloader_setup import dataloader_setup
from validation.best_params.best_params_saver import save_best_params
from validation.grid_search_optimizer import compute_grid_search


def validation(config_settings):
    """
    Method to orchestrate the validation of the model.
    :param config_settings: The configuration settings.
    :return: The new configuration settings.
    """
    # initial message
    info("🔄 Validation started...")

    # set the variable indicating the state of the process
    phase_var.set("validation")

    # load the training set
    training_set, _ = dataloader_setup(
        "training",
        config_settings.training_batch_size,
        False,
        config_settings,
        AccessLogsDataset
    )

    # grid search for best parameters
    best_params = compute_grid_search(
        training_set,
        config_settings
    )

    # set the best parameters and get new config settings
    new_config_settings = save_best_params(
        best_params,
        config_settings
    )

    # print a successful message
    info("✅ Validation successfully completed.")

    return new_config_settings