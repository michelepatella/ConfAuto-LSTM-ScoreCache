from data_preprocessing.cleaner import _remove_missing_values
from data_preprocessing.normalizer import _standardize
from utils.log_utils import _info, phase_var
from utils.config_utils import _get_config_value
from utils.data_utils import _save_dataset, _load_dataset, _get_dataset_path_type


def data_preprocessing():
    """
    Method to orchestrate data preprocessing of dataset.
    :return:
    """
    # initial message
    _info("🔄 Data preprocessing started...")

    # set the variable indicating the state of the process
    phase_var.set("data_preprocessing")

    # load the dataset
    df = _load_dataset()

    # remove missing values
    df_no_missing_values = _remove_missing_values(df)

    # standardize 'delta_time' column
    df_standardized = _standardize(
        df_no_missing_values,
        [
            "id",
            "delta_time",
            "freq_last_10",
            "freq_last_100",
            "freq_last_1000"
        ]
    )

    # save the preprocessed dataset
    _save_dataset(df_standardized)

    # print a successful message
    _info("✅ Data preprocessing successfully completed.")