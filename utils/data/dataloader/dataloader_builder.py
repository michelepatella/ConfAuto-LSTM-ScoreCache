from torch.utils.data import DataLoader
from utils.logs.log_utils import info, debug


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
    except (
            TypeError,
            ValueError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while creating data loader: {e}.")

    # show a successful message
    info("🟢 Data loader created.")

    return loader