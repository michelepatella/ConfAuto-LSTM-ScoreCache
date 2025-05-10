import numpy as np


def _generate_cyclic_time_features(timestamps):
    """
    Method to generate cyclic time features starting from timestamps.
    :param timestamps: The timestamps generated.
    :return: The cyclic time features (hours of the day and days of the week).
    """
    # check if the timestamps is a valid array/list
    if not isinstance(timestamps, (list, np.ndarray)):
        raise TypeError("Timestamps should be a list or a numpy array.")

    # check if all timestamps are positive
    if not np.all(np.array(timestamps) >= 0):
        raise ValueError("Timestamps should be non-negative values.")

    # try to create cyclic time features
    try:

        # create hour of the day and day of the week as fields
        hours_of_day = np.array([timestamp % 24 for timestamp in timestamps])
        days_of_week = np.array([timestamp // 24 % 7 for timestamp in timestamps])

        # add cyclic features
        hour_of_day_sin = np.sin(2 * np.pi * hours_of_day / 24)
        hour_of_day_cos = np.cos(2 * np.pi * hours_of_day / 24)

        day_of_week_sin = np.sin(2 * np.pi * days_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * days_of_week / 7)

        return hour_of_day_sin, hour_of_day_cos, day_of_week_sin, day_of_week_cos

    except Exception as e:
        raise Exception(f"An unexpected error while generating cyclic time features: {e}")