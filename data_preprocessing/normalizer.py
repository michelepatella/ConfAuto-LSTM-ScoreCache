import logging
from sklearn.preprocessing import MinMaxScaler
from utils.config_utils import _get_config_value


def _normalize_timestamps(df):
    """
    Method to normalize the timestamps of dataset.
    :param df: The dataframe to normalize.
    :return: The normalized dataframe.
    """
    # ongoing message
    logging.info("🔄 Dataset normalization started...")

    # load training percentage
    training_perc = _get_config_value("data.training_perc")
    total_len = len(df)

    # calculate the final index of the training set
    train_end_idx = int(training_perc * total_len)

    # initialize the scaler
    scaler = MinMaxScaler()

    try:
        # normalize timestamps of training set
        train_timestamps = (df.loc[:train_end_idx - 1, "timestamp"]
                            .values.reshape(-1, 1))
        scaler.fit(train_timestamps)

        # normalize the whole timestamp column using the fitted scaler
        df["timestamp"] = scaler.transform(
            df["timestamp"].values.reshape(-1, 1)
        )
    except Exception as e:
        raise Exception(f"❌ Error while normalizing the dataset: {e}")

    # print a successful message
    logging.info("🟢 Dataset normalized.")

    return df