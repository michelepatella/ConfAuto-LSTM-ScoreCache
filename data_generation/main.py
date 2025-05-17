from data_generation.frequencies_generator import _generate_last_freq
from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from utils.log_utils import _info, _debug, phase_var
from utils.data_utils import _save_dataset, _get_dataset_path_type, _create_dataframe
import numpy as np


def data_generation():
    """
    Method to orchestrate data generation.
    :return:
    """
    # initial message
    _info("🔄 Data generation started...")

    # set the variable indicating the state of the process
    phase_var.set("data_generation")

    # get the dataset path
    _, distribution_type = _get_dataset_path_type()

    # debugging
    _debug(f"⚙️Type of distribution: {distribution_type}.")

    if distribution_type == "static":
        # generate static requests and delta times
        requests, delta_times = _generate_static_requests()
    else:
        # generate dynamic requests and delta times
        requests, delta_times = _generate_dynamic_requests()

    # generate features (last freq w.r.t. requests)
    freq_columns = _generate_last_freq(requests)

    # create dataframe
    df = _create_dataframe(
        {
            "id": np.arange(len(requests)),
            "delta_time": delta_times,
            **freq_columns,
            "request": requests,
        }
    )

    # save the dataset
    _save_dataset(df)

    # show a successful message
    _info("✅ Data generation successfully completed.")