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
    except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError
    ) as e:
        raise RuntimeError(f"❌ Error while removing missing values from the dataset: {e}.")

    # debugging
    debug(f"⚙️Number of rows with missing values: {initial_len - final_len}.")

    # print a successful message
    info("🟢 Missing values removed.")

    return df