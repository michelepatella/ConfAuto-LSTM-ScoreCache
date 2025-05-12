import logging
from data_generation.cyclic_time_features_generator import _generate_cyclic_time_features
from data_generation.dataset_saver import _save_dataset_to_csv
from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from utils.config_utils import load_config, get_config_value


def data_generation():
    """
    Method to orchestrate the (static or dynamic) data generation.
    :return:
    """
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
    elif distribution_type == "dynamic":
        # generate dynamic requests and timestamps
        requests, timestamps = _generate_dynamic_requests(config)
    else:
        raise ValueError("Unknown distribution type.")

    # generate cyclic time features starting from timestamps
    cyclic_time_features = _generate_cyclic_time_features(timestamps)

    # form the columns of the dataset
    dataset_columns = {
        "timestamps": timestamps,
        **cyclic_time_features,
        "requests": requests,
    }

    if distribution_type == "static":
        # save the static dataset
        _save_dataset_to_csv(
            dataset_columns,
            get_config_value(config, "data.static_dataset_path")
        )
    else:
        # save the dynamic dataset
        _save_dataset_to_csv(
            dataset_columns,
            get_config_value(config, "data.dynamic_dataset_path")
        )

    # show a successful message
    logging.info(f"✅Data generation successfully completed.")