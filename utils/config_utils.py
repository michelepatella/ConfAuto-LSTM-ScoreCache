import yaml
import os
from yaml import YAMLError
from utils.log_utils import info, debug


def _get_config_abs_path():
    """
    Method to get the absolute path of the config file.
    :return: The absolute path of the config file.
    """
    # initial message
    info("🔄 Config file absolute path started...")

    try:
        # define the absolute path of the config file
        path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        abs_path = os.path.abspath(path)
    except (NameError, TypeError, AttributeError, OSError) as e:
        raise RuntimeError(f"❌ Error while getting the config absolute path: {e}.")

    # debugging
    debug(f"⚙️ Absolute path of config file: {abs_path}.")

    # show a successful message
    info("🟢 Config file absolute path obtained.")

    return abs_path


def load_config():
    """
    Method to load the config file.
    :return: The configuration object.
    """
    # initial message
    info("🔄 Config file loading started...")

    # get the abs path of the config file
    config_path = _get_config_abs_path()

    try:
        # load the file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError, YAMLError) as e:
        raise RuntimeError(f"❌ Error while loading config file: {e}.")

    # show a successful message
    info("🟢 Config file loaded.")

    return config


def _merge_config(config, updates):
    """
    Method merge an update config with the original config.
    :param config: The config object.
    :param updates: The updated config object.
    :return: The merged config object.
    """
    # initial message
    info("🔄 Config merging started...")

    #check if the config is None
    if config is None:
        config = {}

    # check if there is something to update
    if not isinstance(updates, dict):
        return config

    try:
        # apply merge recursively
        for key, value in updates.items():
            # debugging
            debug(f"⚙️ Merging (key-value): ({key} - {value}).")

            if isinstance(value, dict) and isinstance(config.get(key), dict):
                _merge_config(config[key], value)
            else:
                config[key] = value
    except (AttributeError, TypeError, KeyError, RecursionError) as e:
        raise RuntimeError(f"❌ Error while merging config file: {e}.")

    # show a successful message
    info("🟢 Config merged.")

    return config


def update_config(config, updated_config):
    """
    Method to update the config file.
    :param config: The config object.
    :param updated_config: The updated config to write.
    :return: The updated config settings.
    """
    from config import prepare_config

    # initial message
    info("🔄 Config file updating started...")

    # get the abs path of the config file
    config_path = _get_config_abs_path()

    # debugging
    debug(f"⚙️ Updated config to be saved: {updated_config}.")

    # merge update configs with config file
    merged_config = _merge_config(
        config,
        updated_config
    )

    try:
        # update the config file
        with open(config_path, "w") as config_file:
            yaml.dump(
                merged_config,
                config_file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as e:
        raise RuntimeError(f"❌ Error while updating the config file: {e}.")

    # show a successful message
    info("🟢 Config file updated.")

    # re-validate config and get the new settings
    new_config_settings = prepare_config()

    return new_config_settings


def get_config_value(config, keys):
    """
    Method to get the config value from the config file.
    :param keys: Requested keys.
    :param config: The config object.
    :return: The config value required.
    """
    # initial message
    info("🔄 Config file reading started...")

    if isinstance(keys, str):
        keys = keys.split(".")

    # get the config file
    value = config

    try:
        # find the requested key
        for key in keys:
            value = value[key]

        # debugging
        debug(f"⚙️ Config value (key-value): ({'.'.join(keys)} - {value}).")

        # show a successful message
        info(f"🟢 {keys} read.")

        return value
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as e:
        raise RuntimeError(f"❌ Error while reading config file: {e}.")