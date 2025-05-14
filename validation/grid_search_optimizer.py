import logging
import itertools
from tqdm import tqdm
from utils.config_utils import _get_config_value
from validation.best_params_updater import _check_and_update_best_params
from validation.time_series_cv import _time_series_cv


def _get_parameter_combination():
    """
    Method to combine the parameters.
    :return: The parameters' combination.
    """
    # initial message
    logging.info("🔄 Parameters' combination started...")

    # get the search space
    search_space = _get_config_value("validation.search_space")

    # use a dictionary
    flat_params = {
        (section, param.replace("_range", "")): values
        for section, section_values in search_space.items()
        for param, values in section_values.items()
    }

    # get all the keys
    keys = flat_params.keys()

    # generate all possible combinations
    combinations = list(itertools.product(
        *flat_params.values()
    ))

    # reconstruct combinations back to nested dicts
    param_combinations = []
    for values in combinations:
        combo_dict = {}
        for (section, param), value in zip(keys, values):
            combo_dict.setdefault(section, {})[param] = value
        param_combinations.append(combo_dict)

    # check the parameters combination calculated
    if not param_combinations:
        raise Exception("❌ No parameter combinations found.")

    # show a successful message
    logging.info("🟢 Parameters combined together.")

    return param_combinations


def _grid_search(training_set):
    """
    Method to perform grid search to find the best parameters.
    :param training_set: The training set.
    :return: The best parameters.
    """
    # initial message
    logging.info("🔄 Grid Search started...")

    # initialize the best parameters and average loss
    best_params = {}
    best_avg_loss = float("inf")

    # get the parameters combination
    param_combinations = _get_parameter_combination()

    # grid search
    with tqdm(
            total=len(param_combinations),
            desc="🔍 Grid Search Progress"
    ) as pbar:
        for params in param_combinations:
   
            # perform the time series CV
            avg_loss = _time_series_cv(
                training_set,
                params
            )

            # check the loss and eventually update the best parameters
            best_avg_loss, best_params = _check_and_update_best_params(
                avg_loss,
                best_avg_loss,
                params,
                best_params
            )

            # update the progress bar
            pbar.update(1)

    # print the best parameters found
    logging.info(f"🏆 Best parameters found: {best_params}")

    # print the best average loss found
    logging.info(f"🏆 Best avg loss: {best_avg_loss}")

    # show a successful message
    logging.info("🟢 Grid Search completed.")

    return best_params