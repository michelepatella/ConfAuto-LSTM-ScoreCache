from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from utils.graph_utils import plot_zipf_loglog, plot_key_usage_heatmap, plot_daily_profile
from utils.log_utils import info, debug, phase_var
from utils.dataset_utils import save_dataset, create_dataframe


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
        requests, timestamps = _generate_static_requests(
            config_settings
        )
    else:
        # generate dynamic requests and timestamps
        requests, timestamps = _generate_dynamic_requests(
            config_settings
        )

    # create dataframe
    df = create_dataframe(
        {
            "timestamp": timestamps[:len(requests)],
            "request": requests,
        }
    )

    # save the dataset
    save_dataset(df, config_settings)

    # show some plots
    plot_zipf_loglog(requests)
    plot_daily_profile(timestamps)
    plot_key_usage_heatmap(
        requests,
        timestamps,
        config_settings
    )

    # show a successful message
    info("✅ Data generation successfully completed.")