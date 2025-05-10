import yaml
import logging


def load_config(config_path="../config.yaml"):
    """
    Method to load the config file.
    :param config_path: Path of the config file.
    :return: The configuration object as output.
    """
    # try to load the file
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"An unexpected error while loading config file: {e}")
        return None

    return config