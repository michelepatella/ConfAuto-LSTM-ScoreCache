from sklearn.preprocessing import StandardScaler
from utils.log_utils import info, debug


def _standardize(df, columns, config_settings):
    """
    Method to standardize the specified columns of dataset.
    :param df: The dataframe to standardize.
    :param columns: The columns to standardize.
    :param config_settings: The configuration settings.
    :return: The standardized dataframe.
    """
    # initial message
    info("🔄 Dataset standardization started...")

    total_len = len(df)

    # calculate the final index of the training set
    train_end_idx = int(config_settings.training_perc * total_len)

    # debugging
    debug(f"⚙️Total rows: {total_len}.")
    debug(f"⚙️Training %: {config_settings.training_perc}.")
    debug(f"⚙️Training dataset range (from-to): (0 - {train_end_idx - 1}).")

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
    except (KeyError, AttributeError, TypeError, ValueError, IndexError) as e:
        raise RuntimeError(f"❌ Error while standardizing the dataset: {e}.")

    # debugging
    debug(f"⚙️Normalized columns: {columns}.")

    # print a successful message
    info("🟢 Dataset standardized.")

    return df