from data_preprocessing.features_engineering.encoder import encode_time_trigonometrically
from utils.logs.log_utils import info


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
        df = encode_time_trigonometrically(
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