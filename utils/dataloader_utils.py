import torch
from torch.utils.data import DataLoader
from utils.log_utils import info, debug


def create_data_loader(
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
    info("🔄 Data loader creation started...")

    # debugging
    debug(f"⚙️ Batch size: {batch_size}.")
    debug(f"⚙️ Shuffle: {shuffle}.")

    try:
        # define the loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    except (TypeError, ValueError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while creating data loader: {e}.")

    # show a successful message
    info("🟢 Data loader created.")

    return loader


def dataloader_setup(
        loader_type,
        batch_size,
        shuffle,
        config_settings,
        AccessLogsDataset
):
    """
    Method to prepare the data loader for the training and testing.
    :param loader_type: The loader type ("training" or "testing").
    :param batch_size: The batch size to use.
    :param shuffle: Whether to shuffle the data.
    :param config_settings: The configuration settings.
    :param AccessLogsDataset: The class AccessLogsDataset.
    :return: The created data loader and the corresponding dataset.
    """
    # initial message
    info("🔄 Load setup started...")

    # debugging
    debug(f"⚙️ Loader type: {loader_type}.")
    debug(f"⚙️ Shuffle: {shuffle}.")

    try:
        # get the dataset
        dataset = AccessLogsDataset(
            loader_type,
            config_settings
        )

        # create the data loader starting from the dataset
        loader = create_data_loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    except (FileNotFoundError, IOError, OSError, ValueError, TypeError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while set upping the loader: {e}.")

    # show a successful message
    info("🟢 Loader setup completed.")

    return dataset, loader


def extract_targets_from_dataloader(data_loader):
    """
    Method to extract the targets from the data loader.
    :param data_loader: The data loader from which to extract the targets.
    :return: All the extracted targets.
    """
    # initial message
    info("🔄 Target extraction from loader started...")

    try:
        all_targets = []
        # extract targets from data loader
        for _, _, targets in data_loader:
            all_targets.append(targets - 1)
    except (TypeError, ValueError, IndexError) as e:
        raise RuntimeError(f"❌ Error while extracting targets from loader: {e}.")

    # debugging
    debug(f"⚙️ Target extracted: {all_targets}.")

    # show a successful message
    info("🟢 Target extracted from loader.")

    return torch.cat(all_targets)