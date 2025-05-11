import logging
from utils.config_utils import load_config, update_config, get_config_value


def _save_best_params(best_params):
    """
    Method to save the best parameters to a config file.
    :param best_params: Best parameters found.
    :return:
    """
    # load config file
    config = load_config()

    # try to update the config file
    try:
        # update the best params
        get_config_value(config, "model").update({
            "hidden_size": best_params["hidden_size"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"]
        })
        get_config_value(config, "training").update({
            "learning_rate": best_params["learning_rate"],
        })
    except Exception as e:
        raise Exception(f"An unexpected error while updating config: {e}")

    # update the best parameters on the config file
    update_config(config)
    logging.info(f"Best parameters have been saved on the config file.")