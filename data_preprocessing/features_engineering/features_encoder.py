import numpy as np
from utils.logs.log_utils import info


def encode_time_trigonometrically(
        df,
        time_column
):
    """
    Replace the time column with two new columns representing
    the time of day encoded trigonometrically (sin and cos).
    :param df: The dataframe to process.
    :param time_column: The name of the timestamp column.
    :return: The augmented dataframe.
    """
    # show initial message
    info("🔄 Encoding time column trigonometrically started...")

    period = 24

    try:
        # normalize time to [0, 2pi]
        time_in_cycle = (df[time_column] % period) / period
        angles = time_in_cycle * 2 * np.pi

        # compute sin and cos
        df['sin_time'] = np.sin(angles)
        df['cos_time'] = np.cos(angles)

        # drop original time column
        df = df.drop(columns=[time_column])

    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")

    # show successful message
    info("🟢 Time column encoded trigonometrically.")

    return df