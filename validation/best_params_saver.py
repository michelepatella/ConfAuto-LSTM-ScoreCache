from utils.log_utils import _info, _debug
from utils.config_utils import _load_config, _update_config


def _save_best_params(best_params):
    """
    Method to save the best parameters to a config file.
    :param best_params: Best parameters found.
    :return:
    """
    # initial message
    _info("🔄 Best parameter saving started...")

    # debugging
    _debug(f"⚙️ Best params to save: {best_params}.")

    # load config file
    config = _load_config()

    try:
        for section, params in best_params.items():
            if section not in config:
                raise KeyError(f"❌ Section '{section}' not found in config.")
            config[section].update(params)
    except Exception as e:
        raise Exception(f"❌ Error while saving the best parameters: {e}")

    # update the best parameters on the config file
    _update_config(config)

    # show a successful message
    _info("🟢 Best parameters saved.")