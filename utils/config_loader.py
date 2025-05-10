import yaml
import os


def load_config():
    """
    Method to load the config file.
    :return: The configuration object as output.
    """
    # determine the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "config.yaml")

    # try to load the file
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"An unexpected error while loading config file: {e}")

    return config