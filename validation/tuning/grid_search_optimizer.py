from tqdm import tqdm
from utils.logs.log_utils import info, debug
from validation.best_params.best_params_updater import check_and_update_best_params
from validation.tuning.time_series_cv import compute_time_series_cv
from validation.search_space.combinator import get_parameters_combination


def compute_grid_search(
        training_set,
        config_settings
):
    """
    Method to compute grid search to find the best parameters.
    :param training_set: The training set.
    :param config_settings: The configuration settings.
    :return: The best parameters.
    """
    # initial message
    info("🔄 Grid Search started...")

    # debugging
    debug(f"⚙️ Training set size: {len(training_set)}.")

    # initialize the best parameters and average loss
    best_params = {}
    best_avg_loss = float("inf")

    # get the parameters combination
    param_combinations = get_parameters_combination(
        config_settings
    )

    # grid search
    with tqdm(
        total=len(param_combinations),
        desc="🔍 Grid Search Progress"
    ) as pbar:
        for params in param_combinations:

            # debugging
            debug(f"⚙️ Evaluating combination: {params}")

            # perform the time series CV
            avg_loss = compute_time_series_cv(
                training_set,
                params,
                config_settings
            )

            # check the avg loss and eventually update the best parameters
            best_avg_loss, best_params = check_and_update_best_params(
                avg_loss,
                best_avg_loss,
                params,
                best_params
            )

            # update the progress bar
            pbar.update(1)

    # print the best parameters found
    info(f"🏆 Best parameters found: {best_params}")

    # print the best average loss
    info(f"🏆 Best avg loss found: {best_avg_loss}")

    # show a successful message
    info("🟢 Grid Search completed.")

    return best_params