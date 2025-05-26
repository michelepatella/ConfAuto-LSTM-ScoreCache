import numpy as np
from utils.log_utils import info


def _build_temporal_features(df, time_column, config_settings):
    """
    Method to build temporal features.
    :param df: The dataframe to process.
    :param time_column: The name of the time column.
    :param config_settings: The configuration settings.
    :return: The augmented dataframe.
    """
    # show initial message
    info("🔄 Building temporal features started...")

    period = 24
    # extract timestamp values
    timestamps = df[time_column].values

    # create new features
    delta_t = np.diff(timestamps, prepend=timestamps[0])
    hours = timestamps % period
    is_peak = ((hours >= config_settings.burst_hour_start) &
               (hours <= config_settings.burst_hour_end)).astype(float)

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

    except KeyError as e:
        raise RuntimeError(f"❌ Error during trigonometrical encoding: {e}.")

    # show successful message
    info("🟢 Time column encoded trigonometrically.")

    return df


def build_features(
        df,
        time_column,
        target_column,
        config_settings
):
    """
    Method to orchestrate feature engineering.
    :param df: The original dataframe.
    :param time_column: The time column.
    :param target_column: The target column.
    :param config_settings: The configuration settings.
    :return: The final dataframe.
    """
    # show initial message
    info("🔄 Feature engineering started...")

    try:
        # build new features
        df = _build_temporal_features(df, time_column, config_settings)
        df = _encode_time_trigonometrically(df, time_column)

        # reorder column s.t. target column is the last one
        features = [col for col in df.columns if col != target_column]
        df = df[features + [target_column]]

        # show successful message
        info("🟢 Feature engineering completed.")

        return df
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise RuntimeError(f"❌ Error during feature engineering: {e}.")