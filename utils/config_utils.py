import yaml
import os
from yaml import YAMLError
from utils.log_utils import _info, _debug


def _get_config_abs_path():
    """
    Method to get the absolute path of the config file.
    :return: The absolute path of the config file.
    """
    try:
        # define the absolute path of the config file
        path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        abs_path = os.path.abspath(path)
    except (NameError, TypeError, AttributeError, OSError) as e:
        raise RuntimeError(f"❌ Error while getting the config absolute path: {e}.")

    # debugging
    _debug(f"⚙️ Absolute path of config file: {abs_path}.")

    return abs_path


def _load_config():
    """
    Method to load the config file.
    :return: The configuration object.
    """
    # initial message
    _info("🔄 Config file loading started...")

    # get the abs path of the config file
    config_path = _get_config_abs_path()

    try:
        # load the file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError, YAMLError) as e:
        raise RuntimeError(f"❌ Error while loading config file: {e}.")

    # show a successful message
    _info("🟢 Config file loaded.")

    return config


def _merge_config(config, updates):
    """
    Method merge an update config with the original config.
    :param config: The config object.
    :param updates: The updated config object.
    :return: The merged config object.
    """
    # initial message
    _info("🔄 Config merging started...")

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
            _debug(f"⚙️ Merging (key-value): ({key} - {value}).")

            if isinstance(value, dict) and isinstance(config.get(key), dict):
                _merge_config(config[key], value)
            else:
                config[key] = value
    except (AttributeError, TypeError, KeyError, RecursionError) as e:
        raise RuntimeError(f"❌ Error while merging config file: {e}.")

    # show a successful message
    _info("🟢 Config merged.")

    return config


def _update_config(updated_config):
    """
    Method to update the config file.
    :param updated_config: The updated config to write.
    :return:
    """
    from main import config_settings
    # initial message
    _info("🔄 Config file updating started...")

    # get the abs path of the config file
    config_path = _get_config_abs_path()

    # debugging
    _debug(f"⚙️ Updated config to be saved: {updated_config}.")

    # merge update configs with config file
    merged_config = _merge_config(
        config_settings.config,
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
    _info("🟢 Config file updated.")


def _get_config_value(config, keys):
    """
    Method to get the config value from the config file.
    :param keys: Requested keys.
    :param config: The config object.
    :return: The config value required.
    """
    # initial message
    _info("🔄 Config file reading started...")

    if isinstance(keys, str):
        keys = keys.split(".")

    # get the config file
    value = config

    try:
        # find the requested key
        for key in keys:
            value = value[key]

        # debugging
        _debug(f"⚙️ Config value (key-value): ({'.'.join(keys)} - {value}).")

        # show a successful message
        _info(f"🟢 {keys} read.")

        return value
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as e:
        raise RuntimeError(f"❌ Error while reading config file: {e}.")


def _flatten_search_space(d, parent_key=()):
    """
    Method to make the search space flatten recursively.
    :param d: The search space dictionary.
    :param parent_key: The key path accumulated so far.
    :return: A list of tuples where each tuple contains
    a key path and its associated list of values.
    """
    try:
        items = []
        for k, v in d.items():
            # extrapolate the name of the parameter
            clean_key = k.replace("_range", "")

            # build the new key (tuple with the name of the parameter)
            new_key = parent_key + (clean_key,)

            # if the new value is another dictionary
            # apply recursively this method
            if isinstance(v, dict):
                items.extend(_flatten_search_space(v, new_key))
            else:
                # convert the value to list
                values = v if isinstance(v, list) else [v]

                # add the couple (key, values) to the final list
                items.append((new_key, values))
    except (TypeError, RecursionError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while making flatten the search space: {e}.")

    return items


def _set_nested_dict(d, keys, value):
    """
    Method to set a value in a nested dictionary
    given a list of keys.
    :param d: The dictionary to update.
    :param keys: The list of nested keys.
    :param value: The value to set.
    :return:
    """
    try:
        # current dictionary initialized to the
        # starting dictionary
        current = d

        # iterate over all the keys except the last one
        # to go down the nested levels
        for k in keys[:-1]:

            # if there is not the key, or
            # it is not a dictionary, make it one
            if (k not in current or not
            isinstance(current[k], dict)):
                current[k] = {}

            # go down a level more
            current = current[k]

        # set the desired value in the last position
        # indicate by the sequence
        current[keys[-1]] = value

    except (TypeError, IndexError, KeyError) as e:
        raise RuntimeError(f"❌ Error while setting a value in a nested dictionary: {e}.")