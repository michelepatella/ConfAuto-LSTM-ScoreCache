import logging
import pandas as pd
from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from utils.config_utils import _get_config_value
from utils.dataset_utils import _save_dataset, _get_dataset_path_type


def data_generation():
    """
    Method to orchestrate data generation.
    :return:
    """
    # initial message
    logging.info("🔄 Data generation started...")

    # get the dataset path
    dataset_path, distribution_type = _get_dataset_path_type()

    if distribution_type == "static":
        # generate static requests and timestamps
        requests, timestamps = _generate_static_requests()
    elif distribution_type == "dynamic":
        # generate dynamic requests and timestamps
        requests, timestamps = _generate_dynamic_requests()
    else:
        raise ValueError(f"❌ Invalid distribution type: {distribution_type}")

    # create dataframe
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "request": requests,
        }
    )

    # save the dataset
    _save_dataset(
        df,
        _get_config_value(dataset_path)
    )

    # show a successful message
    logging.info("✅ Data generation successfully completed.")