import numpy as np
from utils.logs.log_utils import info


def generate_temporal_pattern(
        timestamps,
        period,
        config_settings
):
    """
    Method to generate temporal access pattern requests.
    :param timestamps: The timestamps.
    :param period: The period.
    :param config_settings: The configuration settings.
    :return: The delta time generated.
    """
    # initial message
    info("🔄 Temporal access pattern requests generation started...")

    # extract hour of the day
    hour_of_day = (timestamps[-1] % period) / 3600

    # generate a periodic component
    periodic_component = (
            config_settings.periodic_base_scale +
            config_settings.periodic_amplitude *
            np.cos(2 * np.pi * (hour_of_day / 24))
    )

    # generate mid-day burst
    if (
        config_settings.burst_hour_start <= hour_of_day
        <= config_settings.burst_hour_end
    ):
        bursty_scale = config_settings.burst_high
    else:
        bursty_scale = config_settings.burst_low

    # combine periodic and bursty scales
    freq_scale = max(
        0.5,
        periodic_component * bursty_scale
    )

    # calculate delta time
    delta_t = np.random.exponential(
        scale=freq_scale
    )

    # show a successful message
    info(f"🟢 Temporal access pattern requests generated.")

    return delta_t