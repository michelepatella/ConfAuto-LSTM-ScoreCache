import logging
from utils.config_utils import _load_config, _update_config, _get_config_value


def _save_best_params(best_params):
    """
    Method to save the best parameters to a config file.
    :param best_params: Best parameters found.
    :return:
    """
    # load config file
    config = _load_config()

    try:
        # update the best params
        _get_config_value("model").update({
            "hidden_size": best_params["hidden_size"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"]
        })
        _get_config_value("training").update({
            "learning_rate": best_params["learning_rate"],
        })
    except Exception as e:
        raise Exception(f"Error while saving the best parameters: {e}")

    # update the best parameters on the config file
    _update_config(config)
    logging.info("Best parameters have been saved on the config file.")