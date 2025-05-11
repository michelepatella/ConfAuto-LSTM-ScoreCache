from torch.utils.data import DataLoader


def _create_data_loader(dataset, batch_size):
    """
    Method to create data loader.
    :param dataset: The dataset to load.
    :param batch_size: The batch size to use.
    :return: The data loader.
    """
    try:
        # define the loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
    except Exception as e:
        raise Exception(f"An unexpected error while creating data loader: {e}")

    return loader