from data_preprocessing.cleaner import _remove_missing_values
from data_preprocessing.feature_engineering import _encode_time_trigonometrically, build_features
from data_preprocessing.normalizer import normalize
from utils.log_utils import info, phase_var
from utils.dataset_utils import save_dataset, load_dataset


def data_preprocessing(config_settings):
    """
    Method to orchestrate data preprocessing of dataset.
    :return:
    """
    # initial message
    info("🔄 Data preprocessing started...")

    # set the variable indicating the state of the process
    phase_var.set("data_preprocessing")

    # load the dataset
    df = load_dataset(config_settings)

    # remove missing values
    df_no_missing_values = _remove_missing_values(df)

    # feature engineering
    df_final = build_features(
        df_no_missing_values,
        "timestamp",
        "request"
    )

    # normalize some just created features
    df_normalized = normalize(
        df_final,
        ["delta_t"],
        config_settings
    )

    # save the preprocessed dataset
    save_dataset(df_normalized, config_settings)

    # print a successful message
    info("✅ Data preprocessing successfully completed.")