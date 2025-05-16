from data_generation.features_generator import _generate_last_freq
from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from utils.log_utils import _info, _debug, phase_var
from utils.config_utils import _get_config_value
from utils.data_utils import _save_dataset, _get_dataset_path_type, _create_dataframe


def data_generation():
    """
    Method to orchestrate data generation.
    :return:
    """
    # initial message
    _info("🔄 Data generation started...")
    phase_var.set("data_generation")

    # get the dataset path
    dataset_path, distribution_type = _get_dataset_path_type()

    # debugging
    _debug(f"⚙️Type of distribution: {distribution_type}.")

    if distribution_type == "static":
        # generate static requests and delta times
        requests, delta_times = _generate_static_requests()
    elif distribution_type == "dynamic":
        # generate dynamic requests and delta times
        requests, delta_times = _generate_dynamic_requests()
    else:
        raise ValueError(f"❌ Invalid distribution type: {distribution_type}")

    # generate features (last freq w.r.t. requests)
    freq_columns = _generate_last_freq(requests)

    # create dataframe
    df = _create_dataframe(
        {
            "delta_time": delta_times,
            **freq_columns,
            "request": requests,
        }
    )

    # save the dataset
    _save_dataset(
        df,
        _get_config_value(dataset_path)
    )

    # show a successful message
    _info("✅ Data generation successfully completed.")