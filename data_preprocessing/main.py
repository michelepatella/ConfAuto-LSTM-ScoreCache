import logging
from data_preprocessing.cleaner import _remove_duplicates, _remove_missing_values
from data_preprocessing.normalizer import _standardize_timestamps
from utils.config_utils import _get_config_value
from utils.dataset_utils import _save_dataset, _load_dataset, _get_dataset_path_type


def data_preprocessing():
    """
    Method to orchestrate data preprocessing of dataset.
    :return:
    """
    # initial message
    logging.info("🔄 Data preprocessing started...")

    # get the dataset path
    dataset_path,_ = _get_dataset_path_type()

    # load the dataset
    df = _load_dataset(_get_config_value(dataset_path))

    # remove duplicates from the dataset
    df_deduplicated = _remove_duplicates(df, "timestamp")

    # remove missing values
    df_no_missing_values = _remove_missing_values(df_deduplicated)

    # standardize timestamps
    df_standardized = _standardize_timestamps(
        df_no_missing_values,
        ["timestamp"]
    )

    # save the preprocessed dataset
    _save_dataset(
        df_standardized,
        _get_config_value(dataset_path)
    )

    # print a successful message
    logging.info("✅ Data preprocessing successfully completed.")