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
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")