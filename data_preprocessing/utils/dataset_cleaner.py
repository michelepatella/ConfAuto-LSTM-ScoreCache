from utils.logs.log_utils import info, debug


def remove_missing_values(df):
    """
    Method to remove missing values from a dataframe.
    :param df: The dataframe to remove missing values from.
    :return: The dataframe with missing values removed.
    """
    # initial message
    info("🔄 Missing values remotion started...")

    try:
        # size of the original dataset
        initial_len = len(df)

        # remove rows with missing values
        df = df.dropna(axis=0, how='any')

        # size of the dataset without missing values
        final_len = len(df)
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # debugging
    debug(f"⚙️Number of rows with missing values: {initial_len - final_len}.")

    # print a successful message
    info("🟢 Missing values removed.")

    return df