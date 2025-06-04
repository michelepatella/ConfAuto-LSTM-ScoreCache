from utils.data.dataloader.dataloader_builder import create_data_loader
from utils.logs.log_utils import info, debug


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
    except (
            FileNotFoundError,
            IOError,
            OSError,
            ValueError,
            TypeError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while set upping the loader: {e}.")

    # show a successful message
    info("🟢 Loader setup completed.")

    return dataset, loader