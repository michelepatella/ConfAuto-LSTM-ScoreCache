import yaml
import os
import logging


def _get_config_abs_path():
    """
    Method to get the absolute path of the config file.
    :return: The absolute path of the config file.
    """
    # define the absolute path of the config file
    path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    abs_path = os.path.abspath(path)

    return abs_path


def _load_config():
    """
    Method to load the config file.
    :return: The configuration object.
    """
    # initial message
    logging.info("🔄 Config file loading started...")

    # get the abs path of the config file
    config_path = _get_config_abs_path()

    try:
        # load the file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"❌ Error while loading config file: {e}")

    # show a successful message
    logging.info("🟢 Config file loaded.")

    return config


def _merge_config(config, updates):
    """
    Method merge an update config with the original config.
    :param config: The config object.
    :param updates: The updated config object.
    :return: The merged config object.
    """
    #check if the config is None
    if config is None:
        config = {}

    # check if there is something to update
    if not isinstance(updates, dict):
        return config

    try:
        # apply merge recursively
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                _merge_config(config[key], value)
            else:
                config[key] = value
    except Exception as e:
        raise Exception(f"❌ Error while merging config file: {e}")
    return config


def _update_config(updated_config):
    """
    Method to update the config file.
    :param updated_config: The updated config to write.
    :return:
    """
    # initial message
    logging.info("🔄 Config file updating started...")

    # get the abs path of the config file
    config_path = _get_config_abs_path()

    # load config file
    config = _load_config()

    # merge update configs with config file
    merged_config = _merge_config(config, updated_config)

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
    except Exception as e:
        raise Exception(f"❌ Error while updating the config file: {e}")

    # show a successful message
    logging.info("🟢 Config file updated.")


def _get_config_value(keys):
    """
    Method to get the config value from the config file.
    :param keys: Requested keys.
    :return: The config value required.
    """
    # initial message
    logging.info("🔄 Config file reading started...")

    if isinstance(keys, str):
        keys = keys.split(".")

    # get the config file
    config = _load_config()
    value = config

    try:
        # find the requested key
        for key in keys:
            value = value[key]

        # show a successful message
        logging.info(f"🟢 {keys} read.")

        return value
    except Exception as e:
        raise Exception(f"❌ Error while reading config file: {e}")