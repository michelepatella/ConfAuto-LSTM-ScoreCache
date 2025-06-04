import copy
import itertools
from utils.logs.log_utils import info, debug
from validation.search_space.flattener import flatten_search_space
from validation.search_space.setter import set_nested_dict


def get_parameters_combination(config_settings):
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
            flat_params = flatten_search_space(
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
                    set_nested_dict(
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