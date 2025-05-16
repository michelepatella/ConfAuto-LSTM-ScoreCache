import logging
from collections import Counter
import pandas as pd
from torch.utils.data import DataLoader
from utils.config_utils import _get_config_value


def _compute_frequency(sequence, window):
    """
    Method to compute the frequency of a specific sequence in a given window.
    :param sequence: The sequence to compute the frequency of.
    :param window: The window within to compute the frequency.
    :return: The frequency of the sequence.
    """
    # initial message
    logging.info("🔄 Frequency sequence counting started...")

    # debugging
    logging.debug(f"⚙️ Sequence for which to count the frequency: {sequence}.")
    logging.debug(f"⚙️ Window: {window}.")

    try:
        # initialize frequency
        freq = []

        # count the frequency of the sequence
        # within the given temporal window
        for i in range(len(sequence)):
            if i < window:
                recent = sequence[:i]
            else:
                recent = sequence[i - window:i]
            count = Counter(recent)
            freq.append(count[sequence[i]])

    except Exception as e:
        raise Exception(f"❌ Error while computing the frequency of sequence: {e}")

    # debugging
    logging.debug(f"⚙️ Frequency computed (sequence-frequency): ({sequence} - {freq}).")

    # show a successful message
    logging.info(f"🟢 Frequency of the sequence counted.")

    return freq


def _create_dataframe(
        columns,
        use_frequency,
        sequence_column,
):
    """
    Method to create a dataframe.
    :param columns: The columns to create the dataframe from.
    :param use_frequency: Whether to compute the frequency or not.
    :param sequence_column: The column used to count the frequency in the windows.
    :return: The dataframe created.
    """
    # initial message
    logging.info("🔄 Dataset creation started...")

    # read configuration
    windows = _get_config_value("data.freq_windows")

    try:

        # create the dataframe
        df = pd.DataFrame()

        if use_frequency and sequence_column in columns:
            # add further columns to the dataframe
            for w in windows:
                col_name = f"freq_last_{w}"
                df[col_name] = _compute_frequency(
                    list(columns[sequence_column]),
                    window=w
                )

        for col_name, col_values in columns.items():
            df[col_name] = col_values

    except Exception as e:
        raise Exception(f"❌ Error while creating the dataframe: {e}")

    # show a successful message
    logging.info(f"🟢 Dataframe created.")

    return df


def _save_dataset(df, dataset_path):
    """
    Method to save the dataset.
    :param df: Dataframe to save.
    :param dataset_path: The path of the dataset to save.
    :return:
    """
    # initial message
    logging.info("🔄 Dataset saving started...")

    # debugging
    logging.debug(f"⚙️ Dataset shape to save: {df.shape}.")

    try:
        # convert dataframe to CSV file
        df.to_csv(dataset_path, index=False)

        # show a successful message
        logging.info(f"🟢 Dataset saved to '{dataset_path}'.")
    except Exception as e:
        raise Exception(f"❌ Error while saving the dataset: {e}")


def _create_data_loader(
        dataset,
        batch_size,
        shuffle=False
):
    """
    Method to create data loader from a dataset.
    :param dataset: The dataset to load.
    :param batch_size: The batch size to use.
    :param shuffle: Whether to shuffle the dataset.
    :return: The data loader.
    """
    # initial message
    logging.info("🔄 Data loader creation started...")

    # debugging
    logging.debug(f"⚙️ Batch size: {batch_size}.")
    logging.debug(f"⚙️ Shuffle: {shuffle}.")

    try:
        # define the loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    except Exception as e:
        raise Exception(f"❌ Error while creating data loader: {e}")

    # show a successful message
    logging.info("🟢 Data loader created.")

    return loader


def _load_dataset(dataset_path):
    """
    Method to load the dataset.
    :param dataset_path: Path of the dataset to load.
    :return: The dataset read.
    """
    # initial message
    logging.info("🔄 Dataset loading started...")

    # debugging
    logging.debug(f"⚙️ Path of the dataset to be loaded: {dataset_path}.")

    try:
        # load the dataset
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise Exception(f"❌ Error while loading dataset: {e}")

    # debugging
    logging.debug(f"⚙️ Shape of the dataset loaded: {df.shape}.")

    # show a successful message
    logging.info("🟢 Dataset loaded.")

    return df


def _get_dataset_path_type():
    """
    Method to get the dataset path and type from config file.
    :return: The dataset path and type.
    """
    # initial message
    logging.info("🔄 Dataset path and type retrieval started...")

    # read the dataset type
    dataset_type = _get_config_value("data.distribution_type")

    # debugging
    logging.debug(f"⚙️ Dataset distribution type from config: {dataset_type}.")

    # keep track of the dataset path
    if dataset_type == "static":
        dataset_path = "data.static_dataset_path"
    elif dataset_type == "dynamic":
        dataset_path = "data.dynamic_dataset_path"
    else:
        raise Exception(f"❌ Invalid dataset type: {dataset_type}")

    # debugging
    logging.debug(f"⚙️ Dataset path found: {dataset_path}.")

    # show a successful message
    logging.info("🟢 Dataset path and type retrieved.")

    return dataset_path, dataset_type