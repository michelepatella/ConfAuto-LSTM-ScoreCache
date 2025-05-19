from data_generation.frequencies_generator import _generate_last_relative_frequency
from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from utils.graph_utils import plot_key_frequencies_histogram, plot_keys_transition_matrix, plot_zipf_loglog, plot_requests_over_time
from utils.log_utils import info, debug, phase_var
from utils.dataset_utils import save_dataset, create_dataframe
import numpy as np


def data_generation(config_settings):
    """
    Method to orchestrate data generation.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Data generation started...")

    # set the variable indicating the state of the process
    phase_var.set("data_generation")

    # debugging
    debug(f"⚙️Type of distribution: {config_settings.distribution_type}.")

    if config_settings.distribution_type == "static":
        # generate static requests and delta times
        requests, delta_times = _generate_static_requests(config_settings)
    else:
        # generate dynamic requests and delta times
        requests, delta_times = _generate_dynamic_requests(config_settings)

    # generate other features (last relative frequencies w.r.t. requests)
    freq_columns = _generate_last_relative_frequency(
        requests,
        config_settings
    )

    # create dataframe
    df = create_dataframe(
        {
            "id": np.arange(len(requests)),
            "delta_time": delta_times,
            **freq_columns,
            "request": requests,
        }
    )

    # save the dataset
    save_dataset(df, config_settings)

    # show some plots
    plot_key_frequencies_histogram(requests)
    plot_keys_transition_matrix(requests)
    plot_zipf_loglog(requests)
    plot_requests_over_time(requests, delta_times)

    # show a successful message
    info("✅ Data generation successfully completed.")