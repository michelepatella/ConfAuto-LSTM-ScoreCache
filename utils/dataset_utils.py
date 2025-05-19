import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from utils.log_utils import _info, _debug


def _create_dataframe(columns):
    """
    Method to create a dataframe.
    :param columns: The columns to create the dataframe from.
    :return: The dataframe created.
    """
    # initial message
    _info("🔄 Dataset creation started...")

    try:
        # check the columns
        if any(col is None for col in columns):
            raise ValueError("❌ One or more elements in 'columns' are None.")

        # create the dataframe
        df = pd.DataFrame(columns)

    except (ValueError, TypeError) as e:
        raise RuntimeError(f"❌ Error while creating the dataframe: {e}.")

    # show a successful message
    _info(f"🟢 Dataframe created.")

    return df


def _save_dataset(df, config_settings):
    """
    Method to save the dataset.
    :param df: Dataframe to save.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    _info("🔄 Dataset saving started...")

    # get the dataset path
    dataset_path = _get_dataset_path_type(config_settings)

    # debugging
    _debug(f"⚙️ Dataset shape to save: {df.shape}.")
    _debug(f"⚙️ Dataset path: {dataset_path}.")

    try:
        # convert dataframe to CSV file
        df.to_csv(dataset_path, index=False)

        # show a successful message
        _info(f"🟢 Dataset saved to '{dataset_path}'.")
    except (OSError, PermissionError, FileNotFoundError, ValueError, TypeError) as e:
        raise RuntimeError(f"❌ Error while saving the dataset: {e}.")


def _get_dataset_path_type(config_settings):
    """
    Method to get the dataset path and type from config file.
    :param config_settings: The configuration settings.
    :return: The dataset path.
    """
    # initial message
    _info("🔄 Dataset path and type retrieval started...")

    # debugging
    _debug(f"⚙️ Dataset distribution type from config: "
           f"{config_settings.distribution_type}.")

    # keep track of the dataset path
    if config_settings.distribution_type == "static":
        dataset_path = config_settings.static_save_path
    else:
        dataset_path = config_settings.dynamic_save_path

    # debugging
    _debug(f"⚙️ Dataset path found: {dataset_path}.")

    # show a successful message
    _info("🟢 Dataset path and type retrieved.")

    return dataset_path


def _load_dataset(config_settings):
    """
    Method to load the dataset.
    :param config_settings: The configuration settings.
    :return: The dataset read.
    """
    # initial message
    _info("🔄 Dataset loading started...")

    # get the dataset path
    dataset_path = _get_dataset_path_type(config_settings)

    # debugging
    _debug(f"⚙️ Path of the dataset to be loaded: {dataset_path}.")

    try:
        # load the dataset
        df = pd.read_csv(dataset_path)
    except (ValueError, EmptyDataError, ParserError, UnicodeDecodeError) as e:
        raise RuntimeError(f"❌ Error while loading dataset: {e}.")

    # debugging
    _debug(f"⚙️ Shape of the dataset loaded: {df.shape}.")

    # show a successful message
    _info("🟢 Dataset loaded.")

    return df
