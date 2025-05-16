import torch
import logging
from torch.utils.data import Dataset
from utils.config_utils import _get_config_value
from utils.dataset_utils import _load_dataset


class AccessLogsDataset(Dataset):

    def _split_dataset(self, dataset_type):
        """
        Method to split the dataset based on the requested type.
        :param dataset_type: The dataset type requested
        ("training" or "testing").
        :return:
        """
        # debugging
        logging.debug(f"⚙️ Splitting type: {dataset_type}.")

        try:
            # define the splitting's index
            split_idx = int(
                len(self.data) * _get_config_value("data.training_perc")
            )
        except Exception as e:
            raise Exception(f"❌ Error while defining the dataset splitting's index: {e}")

        # debugging
        logging.debug(f"⚙️ Split index: {split_idx}.")

        try:
            # split the dataset
            if dataset_type == "training":
                self.data = self.data[:split_idx]
            elif dataset_type == "testing":
                self.data = self.data[split_idx:]
            else:
                raise ValueError(f"❌ Invalid split type: {dataset_type}")
        except Exception as e:
            raise Exception(f"❌ Error while splitting the dataset: {e}")


    def _set_fields(self, data):
        """
        Method to set the fields of the dataset.
        :param data: The data from which the fields are extracted.
        :return:
        """
        try:
            # assuming all the columns are features
            # while the last one is the target to predict
            self.columns = data.columns.tolist()
            self.features = self.columns[:-1]  # tutte tranne target
            self.target = self.columns[-1]

            # debugging
            logging.debug(f"⚙️ Dataset columns: {self.columns}.")
            logging.debug(f"⚙️ Feature(s): {self.features}.")
            logging.debug(f"⚙️ Target: {self.target}.")

        except Exception as e:
            raise Exception(f"❌ Error while reading the dataset columns: {e}")


    def __init__(self, dataset_path, dataset_type):
        """
        Method to instantiate the access logs dataset.
        :param dataset_path: The path of the access logs dataset.
        :param dataset_type: The type of dataset requested ("training" or "testing").
        """
        # load the dataset
        df = _load_dataset(dataset_path)

        # debugging
        logging.debug(f"⚙️ Dataset shape: {df.shape}.")

        # set data
        self.data = df.copy()

        # split the dataset to assign data properly
        self._split_dataset(dataset_type)

        # set the fields of the dataset
        self._set_fields(self.data)


    def __len__(self):
        """
        Method to return the length of the access logs dataset.
        :return: The length of the access logs dataset.
        """
        return len(self.data)


    def __getitem__(self, idx):
        """
        Method to return features and the next value in the sequence.
        :param idx: The index of the access logs dataset.
        :return: Access logs dataset.
        """
        # debugging
        logging.debug(f"⚙️ Getting item at index: {idx}.")

        try:
            # get the row from the dataframe
            row = self.data.iloc[idx]

            # extract features
            x = torch.tensor(row[self.features].values.astype(float), dtype=torch.float)

            # extract target (assuming class indices start from 1 → convert to 0-based)
            y_key = torch.tensor(int(row[self.target]) - 1, dtype=torch.long)

            # debugging
            logging.debug(f"⚙️ Feature vector shape: {x.shape}.")
            logging.debug(f"⚙️ Target (key id): {y_key.item()}.")

        except Exception as e:
            raise Exception(f"❌ Error retrieving item at index {idx}: {e}")

        return x, y_key