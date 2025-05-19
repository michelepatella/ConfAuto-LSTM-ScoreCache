import torch
from torch.utils.data import DataLoader
from utils.config_utils import _get_config_value
from utils.dataset_utils import _get_dataset_path_type
from utils.log_utils import _info, _debug


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
    except (TypeError, ValueError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while creating data loader: {e}.")

    # show a successful message
    _info("🟢 Data loader created.")

    return loader


def _loader_setup(
        loader_type,
        shuffle,
        config_settings
):
    """
    Method to prepare the data loader for the training and testing.
    :param loader_type: The loader type ("training" or "testing").
    :param shuffle: Whether to shuffle the data.
    :param config_settings: The configuration settings.
    :return: The created data loader and the corresponding dataset.
    """
    from utils.AccessLogsDataset import AccessLogsDataset

    # initial message
    _info("🔄 Load setup started...")

    # get the dataset type
    dataset_path = _get_dataset_path_type(config_settings)

    # debugging
    _debug(f"⚙️ Loader type: {loader_type}.")
    _debug(f"⚙️ Shuffle: {shuffle}.")

    try:
        # get the dataset
        dataset = AccessLogsDataset(dataset_path)

        # create the data loader starting from the dataset
        loader = _create_data_loader(
            dataset,
            _get_config_value(
                config_settings.config,
                f"{loader_type}.general.batch_size"
            ),
            shuffle
        )
    except (FileNotFoundError, IOError, OSError, ValueError, TypeError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while set upping the loader: {e}.")

    # show a successful message
    _info("🟢 Loader setup completed.")

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
        for _, _, targets in data_loader:
            all_targets.append(targets - 1)
    except (TypeError, ValueError, IndexError) as e:
        raise RuntimeError(f"❌ Error while extracting targets from loader: {e}.")

    # debugging
    _debug(f"⚙️ Target extracted: {all_targets}.")

    # show a successful message
    _info("🟢 Target extracted from loader.")

    return torch.cat(all_targets)
