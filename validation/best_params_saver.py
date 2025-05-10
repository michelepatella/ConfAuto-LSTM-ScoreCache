import yaml

from utils.config_loader import load_config


def _save_best_params(best_params):
    """
    Method to save the best parameters into a configuration file.
    :param best_params: Best parameters found.
    :return:
    """
    # load configuration
    config = load_config()

    # print the best params found
    print("Best params from Stratified 10-folds CV:", best_params)

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

    # write
    with open("../config.yaml", "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    print(f"Best parameters have been saved.")