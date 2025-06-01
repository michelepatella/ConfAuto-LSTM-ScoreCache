import pandas as pd
from torch.utils.data import Subset
from pandas.errors import EmptyDataError, ParserError
from utils.log_utils import info, debug


def create_dataframe(columns):
    """
    Method to create a dataframe.
    :param columns: The columns to create the dataframe from.
    :return: The dataframe created.
    """
    # initial message
    info("🔄 Dataset creation started...")

    try:
        # check the columns
        if any(col is None for col in columns):
            raise ValueError("❌ One or more elements in 'columns' are None.")

        # create the dataframe
        df = pd.DataFrame(columns)
    except (
            ValueError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while creating the dataframe: {e}.")

    # show a successful message
    info(f"🟢 Dataframe created.")

    return df


def save_dataset(
        df,
        config_settings
):
    """
    Method to save the dataset.
    :param df: Dataframe to save.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Dataset saving started...")

    # get the dataset path
    dataset_path = _get_dataset_path_type(config_settings)

    # debugging
    debug(f"⚙️ Dataset shape to save: {df.shape}.")
    debug(f"⚙️ Dataset path: {dataset_path}.")

    try:
        # convert dataframe to CSV file
        df.to_csv(
            dataset_path,
            index=False
        )

        # show a successful message
        info(f"🟢 Dataset saved to '{dataset_path}'.")
    except (
            OSError,
            PermissionError,
            FileNotFoundError,
            ValueError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while saving the dataset: {e}.")


def _get_dataset_path_type(config_settings):
    """
    Method to get the dataset path and type from config file.
    :param config_settings: The configuration settings.
    :return: The dataset path.
    """
    # initial message
    info("🔄 Dataset path and type retrieval started...")

    # debugging
    debug(f"⚙️ Dataset distribution type from config: "
           f"{config_settings.distribution_type}.")

    # keep track of the dataset path
    if config_settings.distribution_type == "static":
        dataset_path = config_settings.static_save_path
    else:
        dataset_path = config_settings.dynamic_save_path

    # debugging
    debug(f"⚙️ Dataset path found: {dataset_path}.")

    # show a successful message
    info("🟢 Dataset path and type retrieved.")

    return dataset_path


def load_dataset(config_settings):
    """
    Method to load the dataset.
    :param config_settings: The configuration settings.
    :return: The dataset read.
    """
    # initial message
    info("🔄 Dataset loading started...")

    # get the dataset path
    dataset_path = _get_dataset_path_type(config_settings)

    # debugging
    debug(f"⚙️ Path of the dataset to be loaded: {dataset_path}.")

    try:
        # load the dataset
        df = pd.read_csv(dataset_path)
    except (
            ValueError,
            EmptyDataError,
            ParserError,
            UnicodeDecodeError
    ) as e:
        raise RuntimeError(f"❌ Error while loading dataset: {e}.")

    # debugging
    debug(f"⚙️ Shape of the dataset loaded: {df.shape}.")

    # show a successful message
    info("🟢 Dataset loaded.")

    return df


def split_training_set(
        training_set,
        config_settings,
        training_indices=None,
        validation_indices=None
):
    """
    Method to split the dataset into training and validation sets.
    :param training_set: The training set to split.
    :param config_settings: The configuration settings.
    :param training_indices: Validation set index.
    :param validation_indices: Training set index.
    :return: The training and validation sets.
    """
    try:
        if (
            training_indices is None or
            validation_indices is None
        ):
            # calculate total training set size
            total_training_size = len(training_set)

            # calculate training and validation size
            training_size = int(
                (1.0 - config_settings.validation_perc) *
                total_training_size
            )
            validation_size = int(
                config_settings.validation_perc *
                total_training_size
            )

            # create indexes for training and validation
            training_indices = list(range(
                0,
                training_size
            ))
            validation_indices = list(range(
                training_size,
                training_size + validation_size
            ))

        # split the training set into training and validation set
        final_training_set = Subset(
            training_set,
            training_indices
        )
        final_validation_set = Subset(
            training_set,
            validation_indices
        )
    except (
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
        NameError
    ) as e:
        raise RuntimeError(f"❌ Error while splitting training set into training and validation sets: {e}.")

    return (
        final_training_set,
        final_validation_set
    )