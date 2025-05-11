import numpy as np
from data_generation.dataset_saver import _save_dataset_to_csv
from data_generation.requests_generator import _generate_static_requests, _generate_dynamic_requests
from data_generation.cyclic_time_features_generator import _generate_cyclic_time_features
from utils.config_utils import load_config, get_config_value


def _generate_static_dataset():
    """
    Method to generate static dataset following Zipf's distribution.
    :return:
    """
    # load config file
    config = load_config()
    num_requests = get_config_value(config, "data.num_requests")
    num_keys = get_config_value(config, "data.num_keys")
    alpha = get_config_value(config, "data.alpha")

    # generate static requests and timestamps
    requests, timestamps = _generate_static_requests(
        num_requests,
        num_keys,
        alpha
    )

    # generate cyclic time features starting from timestamps
    (hour_of_day_sin,
     hour_of_day_cos,
     day_of_week_sin,
     day_of_week_cos) = _generate_cyclic_time_features(timestamps)

    # save the static dataset
    _save_dataset_to_csv(
        timestamps,
        hour_of_day_sin,
        hour_of_day_cos,
        day_of_week_sin,
        day_of_week_cos,
        requests,
        get_config_value(config, "data.static_dataset_path")
    )


def _generate_dynamic_dataset():
    """
    Method to generate dynamic dataset following Zipf's distribution.
    :return:
    """
    # load config file
    config = load_config()
    num_requests = get_config_value(config, "data.num_requests")
    num_keys = get_config_value(config, "data.num_keys")
    alpha_start = get_config_value(config, "data.alpha_start")
    alpha_end = get_config_value(config, "data.alpha_end")
    time_steps = get_config_value(config, "data.time_steps")

    # generate the Zipf distribution's parameter values
    alpha_values = np.linspace(alpha_start, alpha_end, time_steps)

    # generate dynamic requests and timestamps
    requests, timestamps = _generate_dynamic_requests(
        num_requests,
        num_keys,
        alpha_values,
        time_steps
    )

    # generate cyclic time features starting from timestamps
    (hour_of_day_sin,
     hour_of_day_cos,
     day_of_week_sin,
     day_of_week_cos) = _generate_cyclic_time_features(timestamps)

    # save the static dataset
    _save_dataset_to_csv(
        timestamps,
        hour_of_day_sin,
        hour_of_day_cos,
        day_of_week_sin,
        day_of_week_cos,
        requests,
        get_config_value(config, "data.dynamic_dataset_path")
    )