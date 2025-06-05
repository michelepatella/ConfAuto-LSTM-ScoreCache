from data_preprocessing.features_engineering.features_builder import build_features
from data_preprocessing.utils.dataset_cleaner import remove_missing_values
from utils.logs.log_utils import info, phase_var
from utils.data.dataset.dataset_loader import load_dataset
from utils.data.dataset.dataset_saver import save_dataset


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
    df_no_missing_values = remove_missing_values(df)

    # feature engineering
    df_final = build_features(
        df_no_missing_values,
        "timestamp",
        "request",
    )

    # save the preprocessed dataset
    save_dataset(
        df_final,
        config_settings
    )

    # print a successful message
    info("✅ Data preprocessing successfully completed.")