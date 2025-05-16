import logging


def _remove_missing_values(df):
    """
    Method to remove missing values from a dataframe.
    :param df: The dataframe to remove missing values from.
    :return: The dataframe with missing values removed.
    """
    # initial message
    logging.info("🔄 Missing values remotion started...")

    # size of the original dataset
    initial_len = len(df)

    try:
        # remove rows with missing values
        df = df.dropna(axis=0, how='any')
    except Exception as e:
        raise Exception(f"❌ Error while removing missing values from the dataset: {e}")

    # size of the dataset without missing values
    final_len = len(df)

    # debugging
    logging.debug(f"⚙️Number of rows with missing values: {initial_len - final_len}.")

    # print a successful message
    logging.info("🟢 Missing values removed.")

    return df