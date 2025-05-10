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
    except Exception as e:
        raise Exception(f"An unexpected error while updating config: {e}")

    # try to write the best parameters to the config file
    try:
        config_file_path = "../config.yaml"
        with open(config_file_path, "w") as config_file:
            yaml.dump(config, config_file, default_flow_style=False)
        logging.info(f"Best parameters have been saved.")
    except Exception as e:
        raise Exception(f"An unexpected error while writing the best parameters to the config file: {e}")