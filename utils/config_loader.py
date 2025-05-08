import yaml

def load_config(config_path="../config.yaml"):
    """
    Method to load the config file.
    :param config_path: Path of the config file
    :return: The configuration object
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config