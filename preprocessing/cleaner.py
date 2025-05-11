import logging


def _remove_duplicates(df):
    """
    Method to remove duplicate rows from a dataframe.
    :param df: Dataframe to remove duplicate rows from.
    :return: The dataframe with duplicate rows removed.
    """
    try:
        # clear the dataset removing duplicated timestamps
        df.drop_duplicates(subset=['timestamp'], inplace=True)
    except Exception as e:
        raise Exception(f"An unexpected error while deduplicating the dataset: {e}")

    # print a successful message
    logging.info("Dataset correctly deduplicated.")

    return df