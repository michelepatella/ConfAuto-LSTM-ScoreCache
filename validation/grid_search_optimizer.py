import copy
import logging
import itertools
from tqdm import tqdm
from utils.config_utils import _get_config_value
from validation.best_params_updater import _check_and_update_best_params
from validation.time_series_cv import _time_series_cv


def _flatten_search_space(d, parent_key=()):
    """
    Method to make the search space flatten recursively.
    :param d: The search space dictionary.
    :param parent_key: The key path accumulated so far.
    :return: A list of tuples where each tuple contains
    a key path and its associated list of values.
    """
    items = []
    for k, v in d.items():
        clean_key = k.replace("_range", "")
        new_key = parent_key + (clean_key,)
        if isinstance(v, dict):
            items.extend(_flatten_search_space(v, new_key))
        else:
            values = v if isinstance(v, list) else [v]
            items.append((new_key, values))
    return items


def _set_nested_dict(d, keys, value):
    """
    Method to set a value in a nested dictionary
    given a list of keys.
    :param d: The dictionary to update.
    :param keys: The list of nested keys.
    :param value: The value to set.
    :return:
    """
    current = d
    for k in keys[:-1]:
        if (k not in current or not
        isinstance(current[k], dict)):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def _get_parameter_combination():
    """
    Method to combine the parameters.
    :return: The parameters' combination.
    """
    # initial message
    logging.info("🔄 Parameters' combination started...")

    # get the search space
    search_space = _get_config_value("validation.search_space")

    section_combinations = []
    # iterate over all the sections in the search space
    for section, params_dict in search_space.items():
        # make the sections flatten
        flat_params = _flatten_search_space(params_dict)

        keys = [key for key, _ in flat_params]
        value_lists = [v for _, v in flat_params]

        # get the combinations
        combinations = list(itertools.product(*value_lists))
        section_values = []
        for values in combinations:
            combo = {}
            for key_path, value in zip(keys, values):
                _set_nested_dict(combo, key_path, value)
            section_values.append(combo)

        # store the combinations
        section_combinations.append((section, section_values))

    # get the final combinations
    all_combos = list(itertools.product(
        *[vals for _, vals in section_combinations]
    ))

    # reconstruct nested dictionaries
    param_combinations = []
    for combo in all_combos:
        full_dict = {}
        for (section, _), section_dict in zip(section_combinations, combo):
            full_dict[section] = copy.deepcopy(section_dict)

        param_combinations.append(full_dict)

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