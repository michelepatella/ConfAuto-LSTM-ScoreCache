import yaml
from config.config_io.config_locator import get_config_abs_path
from config.config_io.config_loader import load_config
from utils.logs.log_utils import info, debug


def _merge_config(
        config,
        updates
):
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

            if (
                isinstance(value, dict) and
                isinstance(config.get(key), dict)
            ):
                _merge_config(config[key], value)
            else:
                config[key] = value
    except (
            AttributeError,
            TypeError,
            KeyError,
            RecursionError
    ) as e:
        raise RuntimeError(f"❌ Error while merging config file: {e}.")

    # show a successful message
    info("🟢 Config merged.")

    return config


def update_config(
        updated_config,
        prepare_config
):
    """
    Method to update the config file.
    :param prepare_config: Method to validate the config and
    return the updated config settings.
    :param updated_config: The updated config to write.
    :return: The updated config settings.
    """
    # initial message
    info("🔄 Config file updating started...")

    # get the abs path of the config file
    config_path = get_config_abs_path()

    # debugging
    debug(f"⚙️ Updated config to be saved: {updated_config}.")

    # load the original file
    original_config = load_config()

    # merge update configs with config file
    merged_config = _merge_config(
        original_config,
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
    except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            OSError
    ) as e:
        raise RuntimeError(f"❌ Error while updating the config file: {e}.")

    # show a successful message
    info("🟢 Config file updated.")

    # re-validate config and get the new settings
    new_config_settings = prepare_config()

    return new_config_settings