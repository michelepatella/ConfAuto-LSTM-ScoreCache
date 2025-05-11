import logging
from sklearn.preprocessing import MinMaxScaler
from utils.config_utils import load_config, get_config_value


def _normalize_timestamps(df):
    """
    Method to normalize the timestamps of dataset.
    :param df: The dataset to normalize.
    :return: The normalized dataset.
    """
    # load config file
    config = load_config()

    # try to load training percentage from data config
    try:
        # load percentage
        training_perc = get_config_value(config, "data.training_perc")
        total_len = len(df)
    except Exception as e:
        raise Exception(f"An unexpected error while loading data configuration: {e}")

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
        raise Exception(f"An unexpected error while normalizing timestamps: {e}")

    # print a successful message
    logging.info("Timestamps correctly normalized.")

    return df