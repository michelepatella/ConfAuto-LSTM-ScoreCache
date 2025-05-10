import logging
import yaml
from utils.config_loader import load_config


def _save_best_params(best_params):
    """
    Method to save the best parameters to a config file.
    :param best_params: Best parameters found.
    :return:
    """
    # load config file
    config = load_config()

    # check config file read
    if config is None:
        raise ValueError("Error while loading or reading config file.")

    # try to update the config file
    try:
        # update the best params
        config["model"].update({
            "embedding_dim": best_params["embedding_dim"],
            "hidden_size": best_params["hidden_size"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"]
        })
        config["training"].update({
            "learning_rate": best_params["learning_rate"],
        })
    except KeyError as e:
        logging.error(f"Missing key in current parameters: {e}")
    except Exception as e:
        logging.error(f"An unexpected error while updating config: {e}")
        return

    # try to write the best parameters to the config file
    try:
        config_file_path = "../config.yaml"
        with open(config_file_path, "w") as config_file:
            yaml.dump(config, config_file, default_flow_style=False)
        logging.info(f"Best parameters have been saved.")
    except FileNotFoundError as e:
        logging.error(f"Error opening the file: {e}")
    except yaml.YAMLError as e:
        logging.error(f"Error writing the YAML file: {e}")
    except Exception as e:
        logging.error(f"An unexpected error while writing the config file: {e}")