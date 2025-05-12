import logging
from data_preprocessing.cleaner import _remove_duplicates
from data_preprocessing.normalizer import _normalize_timestamps
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
    df_deduplicated = _remove_duplicates(df)

    # normalize timestamps
    df_normalized = _normalize_timestamps(df_deduplicated)

    # save the preprocessed dataset
    _save_dataset(
        df_normalized,
        _get_config_value(dataset_path)
    )

    # print a successful message
    logging.info("✅ Data preprocessing successfully completed.")