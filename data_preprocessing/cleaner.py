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