import copy
import itertools
from tqdm import tqdm
from config.main import search_space
from utils.log_utils import _info, _debug
from utils.config_utils import _flatten_search_space, _set_nested_dict
from validation.best_params_updater import _check_and_update_best_params
from validation.time_series_cv import _time_series_cv


def _get_parameters_combination():
    """
    Method to combine the parameters.
    :return: The parameters' combination.
    """
    # initial message
    _info("🔄 Parameters' combination started...")

    section_combinations = []

    try:
        # iterate over all the sections in the search space
        for section, params_dict in search_space.items():
            # make the sections flatten
            flat_params = _flatten_search_space(params_dict)

            # extrapolate the flatten keys and their values
            keys = [key for key, _ in flat_params]
            value_lists = [v for _, v in flat_params]

            # generate all the possible parameter combinations
            combinations = list(itertools.product(*value_lists))

            section_values = []
            # reconstruct the original nested structure
            # for each combination generated
            for values in combinations:
                combo = {}
                for key_path, value in zip(keys, values):
                    _set_nested_dict(combo, key_path, value)
                section_values.append(combo)

            # store all the combinations
            section_combinations.append((section, section_values))

        # generate all the final combinations
        all_combos = list(itertools.product(
            *[vals for _, vals in section_combinations]
        ))

        # reconstruct the original and complete dictionary
        # for each combination generated
        param_combinations = []
        for combo in all_combos:
            full_dict = {}
            for (section, _), section_dict in zip(section_combinations, combo):
                full_dict[section] = copy.deepcopy(section_dict)

            param_combinations.append(full_dict)

        # debugging
        _debug(f"⚙️ Total parameter combinations found: {len(param_combinations)}")

        # check if there is at least a parameter combination found
        if not param_combinations:
            raise ValueError("❌ No parameters combination found.")

        # show a successful message
        _info("🟢 Parameters combined together.")

        return param_combinations

    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"❌ Error while generating parameter combinations: {e}.")


def _grid_search(training_set):
    """
    Method to perform grid search to find the best parameters.
    :param training_set: The training set.
    :return: The best parameters.
    """
    # initial message
    _info("🔄 Grid Search started...")

    # debugging
    _debug(f"⚙️ Training set size: {len(training_set)}.")

    # initialize the best parameters and average loss
    best_params = {}
    best_avg_loss = float("inf")

    # get the parameters combination
    param_combinations = _get_parameters_combination()

    # grid search
    with tqdm(
            total=len(param_combinations),
            desc="🔍 Grid Search Progress"
    ) as pbar:
        for params in param_combinations:

            # debugging
            _debug(f"⚙️ Evaluating combination: {params}")

            # perform the time series CV
            avg_loss = _time_series_cv(
                training_set,
                params
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
    _info(f"🏆 Best parameters found: {best_params}")

    # print the best average loss
    _info(f"🏆 Best avg loss found: {best_avg_loss}")

    # show a successful message
    _info("🟢 Grid Search completed.")

    return best_params