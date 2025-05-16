import logging
from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from utils.config_utils import _get_config_value
from utils.dataset_utils import _save_dataset, _get_dataset_path_type, _create_dataframe


def data_generation():
    """
    Method to orchestrate data generation.
    :return:
    """
    # initial message
    logging.info("🔄 Data generation started...")

    # get the dataset path
    dataset_path, distribution_type = _get_dataset_path_type()

    # debugging
    logging.debug(f"⚙️Type of distribution: {distribution_type}.")

    if distribution_type == "static":
        # generate static requests and delta times
        requests, delta_times = _generate_static_requests()
    elif distribution_type == "dynamic":
        # generate dynamic requests and delta times
        requests, delta_times = _generate_dynamic_requests()
    else:
        raise ValueError(f"❌ Invalid distribution type: {distribution_type}")

    # create dataframe
    df = _create_dataframe(
        {
            "delta_time": delta_times,
            "request": requests,
        },
        True,
       "request"
    )

    # save the dataset
    _save_dataset(
        df,
        _get_config_value(dataset_path)
    )

    # show a successful message
    logging.info("✅ Data generation successfully completed.")