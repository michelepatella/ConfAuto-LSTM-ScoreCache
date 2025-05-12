import logging
import pandas as pd
from data_generation.cyclic_time_features_generator import _generate_cyclic_time_features
from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from utils.config_utils import load_config, get_config_value
from utils.dataset_utils import _save_dataset


def data_generation():
    """
    Method to orchestrate data generation.
    :return:
    """
    # ongoing message
    logging.info("🔄 Data generation started...")

    # load config file
    config = load_config()

    # read the distribution type
    distribution_type = get_config_value(
        config,
        "data.distribution_type"
    )

    if distribution_type == "static":
        # generate static requests and timestamps
        requests, timestamps = _generate_static_requests(config)

        # keep track of the dataset path
        dataset_path = "data.static_dataset_path"

    elif distribution_type == "dynamic":
        # generate dynamic requests and timestamps
        requests, timestamps = _generate_dynamic_requests(config)

        # keep track of the dataset path
        dataset_path = "data.dynamic_dataset_path"

    else:
        raise ValueError("❌ Unknown distribution type.")

    # generate cyclic time features starting from timestamps
    cyclic_time_features = _generate_cyclic_time_features(timestamps)

    # create dataframe
    df = pd.DataFrame(
        {
            "timestamps": timestamps,
            **cyclic_time_features,
            "requests": requests,
        }
    )

    # save the dataset
    _save_dataset(
        df,
        get_config_value(config, dataset_path)
    )

    # show a successful message
    logging.info("✅ Data generation successfully completed.")