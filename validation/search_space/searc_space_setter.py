from utils.logs.log_utils import info


def set_nested_dict(d, keys, value):
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
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Nested dictionary setting completed.")