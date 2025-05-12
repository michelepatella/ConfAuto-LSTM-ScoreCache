import logging
from data_preprocessing.cleaner import _remove_duplicates
from data_preprocessing.normalizer import _normalize_timestamps
from utils.config_utils import _get_config_value
from utils.dataset_utils import _save_dataset, _load_dataset


def data_preprocessing():
    """
    Method to orchestrate data preprocessing of dataset.
    :return:
    """
    # ongoing message
    logging.info("🔄 Data preprocessing started...")

    # read the dataset type
    dataset_type = _get_config_value("data.distribution_type")

    # keep track of the dataset path
    if dataset_type == "static":
        dataset_path = "data.static_dataset_path"
    elif dataset_type == "dynamic":
        dataset_path = "data.dynamic_dataset_path"
    else:
        raise Exception("❌ Unknown dataset type.")

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