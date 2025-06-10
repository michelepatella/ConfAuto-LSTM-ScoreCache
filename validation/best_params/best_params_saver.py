from config import prepare_config
from utils.logs.log_utils import info, debug
from config.config_io.config_updater import update_config


def save_best_params(
        best_params,
        config_settings
):
    """
    Method to save the best parameters to a config file.
    :param best_params: Best parameters found.
    :param config_settings: Configuration settings.
    :return: The new configuration settings.
    """
    # initial message
    info("🔄 Best parameter saving started...")

    # debugging
    debug(f"⚙️ Best params to save: {best_params}.")

    try:
        # update all the parameters
        for section, params in best_params.items():
            # check section and fields (and subfields, if any)
            if section not in config_settings.config_file:
                raise KeyError(f" Section '{section}' not found in config.")
            if not isinstance(params, dict):
                raise ValueError(f" Parameters for section '{section}' must be a dict. "
                                 f"Received: {type(params)}.")
            if not isinstance(config_settings.config_file[section], dict):
                raise ValueError(f" Config section '{section}' must be a dict. "
                                 f"Found: {type(config_settings.config_file[section])}.")

            # update parameter
            config_settings.config_file[section].update(params)
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # update the best parameters on the config file
    new_config_settings = update_config(
        config_settings.config_file,
        prepare_config
    )

    # show a successful message
    info("🟢 Best parameters saved.")

    return new_config_settings