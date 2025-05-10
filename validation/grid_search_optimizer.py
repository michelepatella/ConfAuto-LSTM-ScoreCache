from tqdm import tqdm
from utils.config_loader import load_config
from validation.best_params_updater import _check_and_update_best_params
from validation.time_series_cv import _time_series_cv


def _grid_search(dataset, criterion):
    """
    Method to perform grid search to find the best parameters.
    :param dataset: The dataset on which to work on.
    :param criterion: The loss function.
    :return: The best parameters as output.
    """
    # load validation configuration
    config = load_config()
    validation_config = config["validation"]

    # initialize the best parameters and average loss
    best_params = {}
    best_avg_loss = float("inf")

    # define the parameters combination
    param_combinations = [
        (embedding_dim, hidden_size, num_layers, dropout, learning_rate)
        for embedding_dim in validation_config["embedding_dim_range"]
        for hidden_size in validation_config["hidden_size_range"]
        for num_layers in validation_config["num_layers_range"]
        for dropout in validation_config["dropout_range"]
        for learning_rate in validation_config["learning_rate_range"]
    ]

    # grid search
    with tqdm(total=len(param_combinations), desc="Grid Search Progress") as pbar:
        for embedding_dim, hidden_size, num_layers, dropout, learning_rate in param_combinations:

            fold_losses = []

            # perform the time series CV
            fold_losses = _time_series_cv(
                dataset,
                embedding_dim,
                hidden_size,
                num_layers,
                dropout,
                learning_rate,
                criterion,
                fold_losses
            )

            # group current parameters together
            curr_params = {
                "embedding_dim": embedding_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": learning_rate
            }

            # check the loss and eventually update the best parameters
            best_avg_loss, best_params = _check_and_update_best_params(
                fold_losses,
                best_avg_loss,
                curr_params,
                best_params
            )

            # update the progress bar
            pbar.update(1)

    # print the best average loss found
    print("Best avg loss:", best_avg_loss)

    return best_params