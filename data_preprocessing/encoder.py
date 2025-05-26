import numpy as np
from utils.log_utils import info


def _encode_time_trigonometrically(
        df,
        time_column,
        period=24
):
    """
    Replace the time column with two new columns representing
    the time of day encoded trigonometrically (sin and cos).
    :param df: The dataframe to process.
    :param time_column: The name of the timestamp column.
    :param period: The period of the cycle.
    :return: The dataframe with added 'sin_time' and 'cos_time' columns,
             and original time_column dropped.
    """
    # show initial message
    info("🔄 Encoding time column trigonometrically started...")

    try:
        # normalize time to [0, 2pi]
        time_in_cycle = (df[time_column] % period) / period
        angles = time_in_cycle * 2 * np.pi

        # compute sin and cos
        df['sin_time'] = np.sin(angles)
        df['cos_time'] = np.cos(angles)

        # drop original time column
        df = df.drop(columns=[time_column])

        # reorder columns: sin_time and cos_time first
        cols = (['sin_time', 'cos_time'] +
                [col for col in df.columns if
                 col not in ['sin_time', 'cos_time']])
        df = df[cols]

    except KeyError as e:
        raise RuntimeError(f"❌ Error during trigonometrical encoding: {e}.")

    # show successful message
    info("🟢 Time column encoded trigonometrically.")

    return df