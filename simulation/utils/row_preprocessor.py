import math
from utils.logs.log_utils import info, debug


def preprocess_data(row):
    """
    Method to process data for the simulation.
    :param row: The current row.
    :return: The preprocessed row.
    """
    # initial message
    info("🔄 Cache simulation preprocessing started...")

    try:
        # get the tuple
        (
            x_features,
            x_keys,
            y_key
        ) = row

        # extract the sin and cos time
        sin_time = x_features[0, 0].item()
        cos_time = x_features[0, 1].item()

        # debugging
        debug(f"⚙️ sin_time: {sin_time}, cos_time: {cos_time}.")

        # calculate timestamps in seconds from info extracted
        angle = math.atan2(
            sin_time,
            cos_time
        )
        if angle < 0:
            angle += 2 * math.pi
        current_time = (
                angle / (2 * math.pi)
                * 24 * 3600
        )

        # extract the key
        key = y_key.item()
    except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            NameError
    ) as e:
        raise RuntimeError(f"❌ Error while preprocessing data for cache simulation: {e}.")

    # print a successful message
    info("🟢 Cache simulation preprocessing completed.")

    return current_time, key