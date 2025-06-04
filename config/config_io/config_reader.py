from utils.logs.log_utils import info, debug


def get_config(
        config,
        keys
):
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
    except (
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
            ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while reading config file: {e}.")