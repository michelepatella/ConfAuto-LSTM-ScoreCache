import copy
import itertools
from tqdm import tqdm
from utils.log_utils import info, debug
from validation.best_params_updater import _check_and_update_best_params
from validation.time_series_cv import _compute_time_series_cv


def _flatten_search_space(d, parent_key=()):
    """
    Method to make the search space flatten recursively.
    :param d: The search space dictionary.
    :param parent_key: The key path accumulated so far.
    :return: A list of tuples where each tuple contains
    a key path and its associated list of values.
    """
    # initial message
    info("🔄 Search space flatting started...")

    try:
        items = []
        for k, v in d.items():
            # extrapolate the name of the parameter
            clean_key = k.replace("_range", "")

            # build the new key (tuple with the name of the parameter)
            new_key = parent_key + (clean_key,)

            # if the new value is another dictionary
            # apply recursively this method
            if isinstance(v, dict):
                items.extend(_flatten_search_space(
                    v,
                    new_key
                ))
            else:
                # convert the value to list
                values = v if isinstance(v, list) else [v]

                # add the couple (key, values) to the final list
                items.append((
                    new_key,
                    values
                ))
    except (
            TypeError,
            RecursionError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while making flatten the search space: {e}.")

    # show a successful message
    info("🟢 Search space flatten.")

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
    # initial message
    info("🔄 Nested dictionary setting started...")

    try:
        # current dictionary initialized to the
        # starting dictionary
        current = d

        # iterate over all the keys except the last one
        # to go down the nested levels
        for k in keys[:-1]:

            # if there is not the key, or
            # it is not a dictionary, make it one
            if (
                k not in current or not
                isinstance(current[k], dict)
            ):
                current[k] = {}

            # go down a level more
            current = current[k]

        # set the desired value in the last position
        # indicate by the sequence
        current[keys[-1]] = value
    except (
            TypeError,
            IndexError,
            KeyError
    ) as e:
        raise RuntimeError(f"❌ Error while setting a value in a nested dictionary: {e}.")

    # show a successful message
    info("🟢 Nested dictionary setting completed.")


def _get_parameters_combination(config_settings):
    """
    Method to combine the parameters.
    :param config_settings: The configuration settings.
    :return: The parameters' combination.
    """
    # initial message
    info("🔄 Parameters' combination started...")

    section_combinations = []

    try:
        # iterate over all the sections in the search space
        for section, params_dict in (
                config_settings.search_space.items()
        ):
            # make the sections flatten
            flat_params = _flatten_search_space(
                params_dict
            )

            # extrapolate the flatten keys and their values
            keys = [key for key, _ in flat_params]
            value_lists = [v for _, v in flat_params]

            # generate all the possible parameter combinations
            combinations = list(itertools.product(
                *value_lists)
            )

            section_values = []
            # reconstruct the original nested structure
            # for each combination generated
            for values in combinations:
                combo = {}
                for key_path, value in zip(
                        keys,
                        values
                ):
                    _set_nested_dict(
                        combo,
                        key_path,
                        value
                    )
                section_values.append(combo)

            # store all the combinations
            section_combinations.append((
                section,
                section_values
            ))

        # generate all the final combinations
        all_combos = list(itertools.product(
            *[vals for _, vals in section_combinations]
        ))

        # reconstruct the original and complete dictionary
        # for each combination generated
        param_combinations = []
        for combo in all_combos:
            full_dict = {}
            for (section, _), section_dict in zip(
                    section_combinations,
                    combo
            ):
                full_dict[section] = copy.deepcopy(
                    section_dict
                )

            param_combinations.append(full_dict)

        # debugging
        debug(f"⚙️ Total parameter combinations found: {len(param_combinations)}")

        # check if there is at least a parameter combination found
        if not param_combinations:
            raise ValueError("❌ No parameters combination found.")

        # show a successful message
        info("🟢 Parameters combined together.")

        return param_combinations

    except (
            KeyError,
            TypeError,
            ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while generating parameter combinations: {e}.")


def _compute_grid_search(training_set, config_settings):
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
    param_combinations = _get_parameters_combination(
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
            avg_loss = _compute_time_series_cv(
                training_set,
                params,
                config_settings
            )

            # check the avg loss and eventually update the best parameters
            best_avg_loss, best_params = _check_and_update_best_params(
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