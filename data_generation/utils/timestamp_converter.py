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
    except NameError as e:
        raise NameError(f"NameError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Timestamps converted.")

    return timestamps