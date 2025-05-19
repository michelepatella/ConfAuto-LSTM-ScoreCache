from utils.log_utils import info, debug


def _remove_duplicates(df, columns):
    """
    Method to remove duplicated rows from a dataframe.
    :param df: Dataframe to remove duplicated rows from.
    :param columns: List of columns to remove duplicated rows from.
    :return: The dataframe with duplicate rows removed.
    """
    # initial message
    info("🔄 Dataset deduplication started...")

    try:
        for column in columns:
            # clear the dataset removing duplicated rows
            df.drop_duplicates(subset=[column], inplace=True)
    except Exception as e:
        raise Exception(f"❌ Error while deduplicating the dataset: {e}")

    # print a successful message
    info("🟢 Dataset deduplicated.")

    return df


def _remove_missing_values(df):
    """
    Method to remove missing values from a dataframe.
    :param df: The dataframe to remove missing values from.
    :return: The dataframe with missing values removed.
    """
    # initial message
    info("🔄 Missing values remotion started...")

    # size of the original dataset
    initial_len = len(df)

    try:
        # remove rows with missing values
        df = df.dropna(axis=0, how='any')
    except (AttributeError, TypeError, ValueError, KeyError) as e:
        raise RuntimeError(f"❌ Error while removing missing values"
                           f" from the dataset: {e}.")

    # size of the dataset without missing values
    final_len = len(df)

    # debugging
    debug(f"⚙️Number of rows with missing values: {initial_len - final_len}.")

    # print a successful message
    info("🟢 Missing values removed.")

    return df