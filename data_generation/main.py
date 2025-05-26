from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from utils.graph_utils import plot_keys_transition_matrix, plot_zipf_loglog, plot_requests_over_time, \
    plot_key_usage_heatmap
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
        # generate static requests and timestamps
        requests, timestamps = _generate_static_requests(config_settings)
    else:
        # generate dynamic requests and timestamps
        requests, timestamps = _generate_dynamic_requests(config_settings)

    # consider timestamps as hours of the day
    timestamps = (timestamps % 24 * 60 * 60) / 3600.0  #

    # create dataframe
    df = create_dataframe(
        {
            "id": np.arange(len(requests)),
            "timestamp": timestamps[:len(requests)],
            "request": requests,
        }
    )

    # save the dataset
    save_dataset(df, config_settings)

    # show some plots
    plot_zipf_loglog(requests)
    plot_keys_transition_matrix(requests)
    plot_requests_over_time(requests, timestamps)
    plot_key_usage_heatmap(requests, timestamps, config_settings)
    print("Stampato")
    # show a successful message
    info("✅ Data generation successfully completed.")