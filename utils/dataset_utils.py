import logging
import pandas as pd
from torch.utils.data import DataLoader


def _save_dataset(df, file_name):
    """
    Method to save the dataset.
    :param df: Dataframe to save.
    :param file_name: The name of the dataset file.
    :return:
    """
    # ongoing message
    logging.info("🔄 Dataset saving started...")

    try:

        # convert dataframe to CSV file
        df.to_csv(file_name, index=False)

        # show a successful message
        logging.info("🟢 Dataset saved to '{file_name}'.")

    except Exception as e:
        raise Exception(f"❌Error while saving the dataset: {e}")


def _create_data_loader(dataset, batch_size):
    """
    Method to create data loader from a dataset.
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
        raise Exception(f"Error while creating data loader: {e}")

    return loader


def _load_dataset(path):
    """
    Method to load the dataset.
    :param path: Path of the dataset.
    :return: The dataset read.
    """
    try:
        # load the dataset
        df = pd.read_csv(path)
    except Exception as e:
        raise Exception(f"❌ Error while reading csv dataset file: {e}")

    return df