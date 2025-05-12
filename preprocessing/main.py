import pandas as pd
import logging
from preprocessing.cleaner import _remove_duplicates
from preprocessing.normalizer import _normalize_timestamps
from preprocessing.preprocessed_dataset_saver import _save_preprocessed_dataset
from utils.config_utils import load_config, get_config_value


def preprocessing(dataset_type):
    """
    Method to orchestrate preprocessing of dataset.
    :param dataset_type: The dataset type to preprocess.
    :return:
    """
    # load config file
    config = load_config()

    # find the dataset path
    if dataset_type == "static":
        dataset_path = "data.static_dataset_path"
    elif dataset_type == "dynamic":
        dataset_path = "data.dynamic_dataset_path"
    else:
        raise Exception("Unknown dataset type passed to preprocessing.")

    try:
        # load the dataset
        df = pd.read_csv(get_config_value(
            config,
            dataset_path
        ))
    except Exception as e:
        raise Exception(f"Error while reading csv dataset file: {e}")

    # remove duplicates from the dataset
    df_deduplicated = _remove_duplicates(df)

    # normalize timestamps
    df_normalized = _normalize_timestamps(df_deduplicated)

    # save the preprocessed dataset
    _save_preprocessed_dataset(
        df_normalized,
        get_config_value(config, dataset_path)
    )

    # print a successful message
    logging.info(f"Preprocessing successfully completed.")