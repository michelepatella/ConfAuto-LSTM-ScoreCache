import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.log_utils import _info, _debug
from utils.config_utils import _get_config_value


def _create_dataframe(columns):
    """
    Method to create a dataframe.
    :param columns: The columns to create the dataframe from.
    :return: The dataframe created.
    """
    # initial message
    _info("🔄 Dataset creation started...")

    try:
        # create the dataframe
        df = pd.DataFrame(columns)
    except Exception as e:
        raise Exception(f"❌ Error while creating the dataframe: {e}")

    # show a successful message
    _info(f"🟢 Dataframe created.")

    return df


def _save_dataset(df, dataset_path):
    """
    Method to save the dataset.
    :param df: Dataframe to save.
    :param dataset_path: The path of the dataset to save.
    :return:
    """
    # initial message
    _info("🔄 Dataset saving started...")

    # debugging
    _debug(f"⚙️ Dataset shape to save: {df.shape}.")

    try:
        # convert dataframe to CSV file
        df.to_csv(dataset_path, index=False)

        # show a successful message
        _info(f"🟢 Dataset saved to '{dataset_path}'.")
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
    _info("🔄 Data loader creation started...")

    # debugging
    _debug(f"⚙️ Batch size: {batch_size}.")
    _debug(f"⚙️ Shuffle: {shuffle}.")

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
    _info("🟢 Data loader created.")

    return loader


def _load_dataset(dataset_path):
    """
    Method to load the dataset.
    :param dataset_path: Path of the dataset to load.
    :return: The dataset read.
    """
    # initial message
    _info("🔄 Dataset loading started...")

    # debugging
    _debug(f"⚙️ Path of the dataset to be loaded: {dataset_path}.")

    try:
        # load the dataset
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise Exception(f"❌ Error while loading dataset: {e}")

    # debugging
    _debug(f"⚙️ Shape of the dataset loaded: {df.shape}.")

    # show a successful message
    _info("🟢 Dataset loaded.")

    return df


def _get_dataset_path_type():
    """
    Method to get the dataset path and type from config file.
    :return: The dataset path and type.
    """
    # initial message
    _info("🔄 Dataset path and type retrieval started...")

    # read the dataset type
    dataset_type = _get_config_value("data.distribution_type")

    # debugging
    _debug(f"⚙️ Dataset distribution type from config: {dataset_type}.")

    # keep track of the dataset path
    if dataset_type == "static":
        dataset_path = "data.static_dataset_path"
    elif dataset_type == "dynamic":
        dataset_path = "data.dynamic_dataset_path"
    else:
        raise Exception(f"❌ Invalid dataset type: {dataset_type}")

    # debugging
    _debug(f"⚙️ Dataset path found: {dataset_path}.")

    # show a successful message
    _info("🟢 Dataset path and type retrieved.")

    return dataset_path, dataset_type


def _loader_setup(loader_type, shuffle):
    """
    Method to prepare the data loader for the training and testing.
    :param loader_type: The loader type ("training" or "testing").
    :param shuffle: Whether to shuffle the data.
    :return: The created data loader and the corresponding dataset.
    """
    from utils.AccessLogsDataset import AccessLogsDataset

    # get the dataset type
    dataset_path, _ = _get_dataset_path_type()

    # debugging
    _debug(f"⚙️ Loader type: {loader_type}.")
    _debug(f"⚙️ Shuffle: {shuffle}.")

    # get the dataset
    dataset = AccessLogsDataset(
        _get_config_value(dataset_path),
        loader_type
    )

    # create the data loader starting from the dataset
    loader = _create_data_loader(
        dataset,
        _get_config_value(f"{loader_type}.batch_size"),
        shuffle
    )

    return dataset, loader


def _extract_targets_from_loader(data_loader):
    """
    Method to extract the targets from the data loader.
    :param data_loader: The data loader from which to extract the targets.
    :return: All the extracted targets.
    """
    # initial message
    _info("🔄 Target extraction from loader started...")

    try:
        all_targets = []
        # extract targets from data loader
        for _, targets in data_loader:
            all_targets.append(targets - 1)

    except Exception as e:
        raise Exception(f"❌ Error while extracting targets from loader: {e}")

    # debugging
    _debug(f"⚙️ Target extracted: {all_targets}.")

    # show a successful message
    _info("🟢 Target extracted from loader.")

    return torch.cat(all_targets)