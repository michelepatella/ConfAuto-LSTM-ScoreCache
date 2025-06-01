import numpy as np
from utils.log_utils import info


def _encode_time_trigonometrically(
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
        raise RuntimeError(f"❌ Error during trigonometrical "
                           f"encoding: {e}.")

    # show successful message
    info("🟢 Time column encoded trigonometrically.")

    return df


def build_features(
        df,
        time_column,
        target_column
):
    """
    Method to orchestrate feature engineering.
    :param df: The original dataframe.
    :param time_column: The time column.
    :param target_column: The target column.
    :return: The final dataframe.
    """
    # show initial message
    info("🔄 Feature engineering started...")

    try:
        # build new features
        df = _encode_time_trigonometrically(
            df,
            time_column
        )

        # reorder column s.t. target column is the last one
        features = [
            col for col in df.columns
            if col != target_column
        ]
        df = df[features + [target_column]]

        # show successful message
        info("🟢 Feature engineering completed.")

        return df
    except (
            KeyError,
            TypeError,
            ValueError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error during feature engineering: {e}.")