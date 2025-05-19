from data_preprocessing.cleaner import _remove_missing_values, _remove_duplicates
from data_preprocessing.standardizer import _standardize
from utils.log_utils import info, phase_var
from utils.dataset_utils import save_dataset, load_dataset


def data_preprocessing(config_settings):
    """
    Method to orchestrate data preprocessing of dataset.
    :return:
    """
    # initial message
    info("🔄 Data preprocessing started...")

    # set the variable indicating the state of the process
    phase_var.set("data_preprocessing")

    # load the dataset
    df = load_dataset(config_settings)

    # deduplicate the dataset
    df_deduplicated = _remove_duplicates(df, ["id"])

    # remove missing values
    df_no_missing_values = _remove_missing_values(df_deduplicated)

    # standardize 'delta_time' column
    df_standardized = _standardize(
        df_no_missing_values,
        [
            "id",
            "delta_time",
            "freq_last_10",
            "freq_last_100",
            "freq_last_1000"
        ],
        config_settings
    )

    # save the preprocessed dataset
    save_dataset(df_standardized, config_settings)

    # print a successful message
    info("✅ Data preprocessing successfully completed.")