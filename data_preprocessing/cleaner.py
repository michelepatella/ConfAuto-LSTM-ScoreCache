import logging


def _remove_duplicates(df):
    """
    Method to remove duplicated rows from a dataframe.
    :param df: Dataframe to remove duplicated rows from.
    :return: The dataframe with duplicate rows removed.
    """
    # initial message
    logging.info("🔄 Dataset deduplication started...")

    try:
        # clear the dataset removing duplicated timestamps
        df.drop_duplicates(subset=['timestamp'], inplace=True)
    except Exception as e:
        raise Exception(f"❌ Error while deduplicating the dataset: {e}")

    # print a successful message
    logging.info("🟢 Dataset deduplicated.")

    return df


def _remove_missing_values(df):
    """
    Method to remove missing values from a dataframe.
    :param df: The dataframe to remove missing values from.
    :return: The dataframe with missing values removed.
    """
    # initial message
    logging.info("🔄 Missing values remotion started...")

    try:
        # remove rows with missing values
        df = df.dropna(axis=0, how='any')
    except Exception as e:
        raise Exception(f"❌ Error while removing missing values from the dataset: {e}")

    # print a successful message
    logging.info("🟢 Missing values removed.")

    return df