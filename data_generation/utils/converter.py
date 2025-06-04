import numpy as np
from utils.logs.log_utils import info


def timestamps_to_hours(timestamps):
    """
    Method to convert timestamps from
    seconds to hours of the day.
    :param timestamps: Timestamps to convert.
    :return: Timestamps as hours of the day.
    """
    # initial message
    info("🔄 Timestamp convertion started...")

    try:
        # consider timestamps as hours of the day
        timestamps = np.array(timestamps) / 3600.0
    except (
        NameError,
        TypeError,
        ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while converting timestamps to hours of the day: {e}.")

    # show a successful message
    info("🟢 Timestamps converted.")

    return timestamps