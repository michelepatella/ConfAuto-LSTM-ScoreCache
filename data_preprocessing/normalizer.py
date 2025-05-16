from sklearn.preprocessing import StandardScaler
from utils.log_utils import _info, _debug
from utils.config_utils import _get_config_value


def _standardize(df, columns):
    """
    Method to standardize the specified columns of dataset.
    :param df: The dataframe to standardize.
    :param columns: The columns to standardize.
    :return: The standardized dataframe.
    """
    # initial message
    _info("🔄 Dataset standardization started...")

    # load training percentage
    training_perc = _get_config_value("data.training_perc")
    total_len = len(df)

    # calculate the final index of the training set
    train_end_idx = int(training_perc * total_len)

    # debugging
    _debug(f"⚙️Total rows: {total_len}.")
    _debug(f"⚙️Training %: {training_perc}.")
    _debug(f"⚙️Training dataset range (from-to): (0 - {train_end_idx - 1}).")

    # initialize the scaler
    scaler = StandardScaler()

    try:
        for column in columns:
            # standardize the specified column
            train_set = (df.loc[:train_end_idx - 1, column]
                                .values.reshape(-1, 1))
            scaler.fit(train_set)

            # standardize the whole specified column using the fitted scaler
            df[column] = scaler.transform(
                df[column].values.reshape(-1, 1)
            )
    except Exception as e:
        raise Exception(f"❌ Error while standardizing the dataset: {e}")

    # debugging
    _debug(f"⚙️Normalized columns: {columns}.")

    # print a successful message
    _info("🟢 Dataset standardized.")

    return df