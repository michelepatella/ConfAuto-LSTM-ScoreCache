import logging
from sklearn.preprocessing import StandardScaler
from utils.config_utils import _get_config_value


def _standardize_timestamps(df):
    """
    Method to standardize the timestamps of dataset.
    :param df: The dataframe to standardize.
    :return: The standardized dataframe.
    """
    # initial message
    logging.info("🔄 Dataset standardization started...")

    # load training percentage
    training_perc = _get_config_value("data.training_perc")
    total_len = len(df)

    # calculate the final index of the training set
    train_end_idx = int(training_perc * total_len)

    # initialize the scaler
    scaler = StandardScaler()

    try:
        # standardize timestamps of training set
        train_timestamps = (df.loc[:train_end_idx - 1, "timestamp"]
                            .values.reshape(-1, 1))
        scaler.fit(train_timestamps)

        # standardize the whole timestamp column using the fitted scaler
        df["timestamp"] = scaler.transform(
            df["timestamp"].values.reshape(-1, 1)
        )
    except Exception as e:
        raise Exception(f"❌ Error while standardizing the dataset: {e}")

    # print a successful message
    logging.info("🟢 Dataset standardized.")

    return df