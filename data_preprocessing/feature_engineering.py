import numpy as np
import pandas as pd
from utils.log_utils import info


def _build_temporal_features(df, time_column):
    """
    Method to build temporal features.
    :param df: The dataframe to process.
    :param time_column: The name of the time column.
    :return: The augmented dataframe.
    """
    # show initial message
    info("🔄 Building temporal features started...")

    period = 24
    # extract timestamp values
    timestamps = df[time_column].values

    # create day fraction, delta time, and
    # a column indicating whether peak or not as
    # three new features
    day_fraction = (timestamps % period) / period
    delta_t = np.diff(timestamps, prepend=timestamps[0])
    hours = timestamps % period
    is_peak = ((hours >= 11) & (hours < 15)).astype(float)

    df["day_fraction"] = day_fraction
    df["delta_t"] = delta_t
    df["is_peak"] = is_peak

    # show successful message
    info("🟢 Temporal features built.")

    return df


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
    :return: The augmented dataframe.
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


def _build_request_features(df, target_column, window_size=5):
    """
    Method to build features related to requests.
    :param df: The dataframe to process.
    :param target_column: The name of the target column.
    :param window_size: The window size.
    :return: The augmented dataframe.
    """
    # show initial message
    info("🔄 Building request features started...")

    # extract target column values
    values = df[target_column].values

    # generate last request, delta key, and moving
    # average key as three new features
    last_request = np.insert(values[:-1], 0, 0)
    delta_key = values - last_request
    moving_avg_key = pd.Series(values).rolling(
        window=window_size,
        min_periods=1
    ).mean().values

    df["last_request"] = last_request
    df["delta_key"] = delta_key
    df["moving_avg_key"] = moving_avg_key

    # show successful message
    info("🟢 Request features built.")

    return df


def build_features(
        df,
        time_column,
        target_column,
        window_size=5
):
    """
    Method to orchestrate feature engineering.
    :param df: The original dataframe.
    :param time_column: The time column.
    :param target_column: The target column.
    :param window_size: The window size.
    :return: The final dataframe.
    """
    # show initial message
    info("🔄 Feature engineering started...")

    try:
        # build new features
        df = _build_temporal_features(df, time_column)
        df = _build_request_features(
            df,
            target_column,
            window_size
        )
        df = _encode_time_trigonometrically(df, time_column)

        # reorder column s.t. target column is the last one
        features = [col for col in df.columns if col != target_column]
        df = df[features + [target_column]]

        # show successful message
        info("🟢 Feature engineering completed.")

        return df
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise RuntimeError(f"❌ Error during feature engineering: {e}.")