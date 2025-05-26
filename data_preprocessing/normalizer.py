from sklearn.preprocessing import MinMaxScaler
from utils.log_utils import info, debug


def normalize(df, columns, config_settings):
    """
    Method to normalize the specified columns of dataset to [0,1].
    :param df: The dataframe to normalize.
    :param columns: The columns to normalize.
    :param config_settings: The configuration settings.
    :return: The normalized dataframe.
    """
    # initial message
    info("🔄 Dataset normalization started...")

    total_len = len(df)

    # calculate the final index of the training set
    train_end_idx = int(config_settings.training_perc * total_len)

    # debugging
    debug(f"⚙️Total rows: {total_len}.")
    debug(f"⚙️Training %: {config_settings.training_perc}.")
    debug(f"⚙️Training dataset range (from-to): (0 - {train_end_idx - 1}).")

    # initialize the scaler
    scaler = MinMaxScaler()

    try:
        for column in columns:
            # normalize the specified column
            train_set = (df.loc[:train_end_idx - 1, column]
                         .values.reshape(-1, 1))
            scaler.fit(train_set)

            # normalize the whole specified column using the fitted scaler
            df[column] = scaler.transform(df[column]
                                          .values.reshape(-1, 1))
    except (KeyError, AttributeError, TypeError, ValueError, IndexError) as e:
        raise RuntimeError(f"❌ Error while normalizing the dataset: {e}.")

    # debugging
    debug(f"⚙️Normalized columns: {columns}.")

    # print a successful message
    info("🟢 Dataset normalized.")

    return df