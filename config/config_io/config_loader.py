import yaml
from yaml import YAMLError
from config.config_io.config_locator import get_config_abs_path
from utils.logs.log_utils import info


def load_config():
    """
    Method to load the config file.
    :return: The configuration object.
    """
    # initial message
    info("🔄 Config file loading started...")

    # get the abs path of the config file
    config_path = get_config_abs_path()

    try:
        # load the file
        with open(config_path, "r") as f:
            config_file = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"FileNotFoundError: {e}.")
    except PermissionError as e:
        raise PermissionError(f"PermissionError: {e}.")
    except IsADirectoryError as e:
        raise IsADirectoryError(f"IsADirectoryError: {e}.")
    except OSError as e:
        raise OSError(f"OSError: {e}.")
    except YAMLError as e:
        raise YAMLError(f"YAMLError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Config file loaded.")

    return config_file