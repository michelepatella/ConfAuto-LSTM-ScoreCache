import math


def preprocess_data(row):
    """
    Method to process data for the simulation.
    :param row: The current row.
    :return: The preprocessed row.
    """
    # get the tuple
    x_features, x_keys, y_key = row

    sin_time = x_features[0, 0].item()
    cos_time = x_features[0, 1].item()

    # calculate timestamps in seconds
    angle = math.atan2(sin_time, cos_time)
    if angle < 0:
        angle += 2 * math.pi
    current_time = angle / (2 * math.pi) * 24 * 3600

    # extract the key
    key = y_key.item()

    return current_time, key