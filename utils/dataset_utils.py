import logging
import pandas as pd
from torch.utils.data import DataLoader


def _save_dataset(df, dataset_path):
    """
    Method to save the dataset.
    :param df: Dataframe to save.
    :param dataset_path: The path of the dataset to save.
    :return:
    """
    # ongoing message
    logging.info("🔄 Dataset saving started...")

    try:

        # convert dataframe to CSV file
        df.to_csv(dataset_path, index=False)

        # show a successful message
        logging.info(f"🟢 Dataset saved to '{dataset_path}'.")

    except Exception as e:
        raise Exception(f"❌ Error while saving the dataset: {e}")


def _create_data_loader(dataset, batch_size):
    """
    Method to create data loader from a dataset.
    :param dataset: The dataset to load.
    :param batch_size: The batch size to use.
    :return: The data loader.
    """
    # ongoing message
    logging.info("🔄 Data loader creation started...")

    try:
        # define the loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
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
    # ongoing message
    logging.info("🔄 Dataset loading started...")

    try:
        # load the dataset
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise Exception(f"❌ Error while loading dataset: {e}")

     # show a successful message
    logging.info("🟢 Dataset loaded.")

    return df