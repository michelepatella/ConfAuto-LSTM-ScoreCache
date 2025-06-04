from utils.logs.log_utils import info


def flatten_search_space(d, parent_key=()):
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
                items.extend(flatten_search_space(
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