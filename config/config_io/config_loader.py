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
    except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            OSError,
            YAMLError
    ) as e:
        raise RuntimeError(f"❌ Error while loading config file: {e}.")

    # show a successful message
    info("🟢 Config file loaded.")

    return config_file