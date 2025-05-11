import yaml
import os


def load_config():
    """
    Method to load the config file.
    :return: The configuration object.
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

def update_config(updated_content):
    """
    Method to update the config file.
    :param updated_content: The updated content to write.
    :return:
    """
    # define the absolute path of the config file
    config_file_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config_file_path = os.path.abspath(config_file_path)

    try:
        # update the config file
        with open(config_file_path, "w") as config_file:
            yaml.dump(
                updated_content, config_file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )
    except Exception as e:
        raise Exception(f"An unexpected error while updating the config file: {e}")

def get_config_value(config, keys):
    """
    Method to get the config value from the config file.
    :param config: Config object.
    :param keys: Requested keys.
    :return: The config value required.
    """
    if isinstance(keys, str):
        keys = keys.split(".")

    value = config
    # try to access to the requested key
    try:
        # find the requested key
        for key in keys:
            value = value[key]

        return value
    except Exception as e:
        raise Exception(f"An unexpected error while reading config file: {e}")