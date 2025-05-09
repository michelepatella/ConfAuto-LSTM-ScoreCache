import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.config_loader import load_config

class AccessLogsDataset(Dataset):

    def __init__(self, csv_path, split):
        """
        Method to instantiate the access logs dataset.
        :param csv_path: The csv path of the access logs dataset.
        :param split: The type of dataset split ("training", "validation", or "testing").
        """
        # load data configuration
        config = load_config()
        data_config = config["data"]

        # get the csv dataset file
        df = pd.read_csv(csv_path)

        # set keys and seq_len fields
        self.keys = df["key"].values
        self.timestamps = df["timestamp"].values
        self.seq_len = data_config["seq_len"]

        # split dataset into training, testing, and validation set
        self.train_size = int(len(self.keys) * data_config["training_perc"])
        self.val_size = int(len(self.keys) * data_config["validation_perc"])
        self.test_size = len(self.keys) - self.train_size - self.val_size

        # split data
        if split == "training":
            self.data = list(zip(
                self.keys[:self.train_size],
                self.timestamps[:self.train_size]
            ))
        elif split == "validation":
            self.data = list(zip(
                self.keys[self.train_size:self.train_size + self.val_size],
                self.timestamps[self.train_size:self.train_size + self.val_size]
            ))
        elif split == "testing":
            self.data = list(zip(
                self.keys[self.train_size + self.val_size:],
                self.timestamps[self.train_size + self.val_size:]
            ))
        else:
            raise ValueError("Invalid split type.")

    def __len__(self):
        """
        Method to return the length of the access logs dataset.
        :return: The length of the access logs dataset as output.
        """
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        """
        Method to return a sequence and the next value in the sequence.
        :param idx: The index of the access logs dataset
        :return: Access logs dataset as output.
        """
        # get the sequence (x)
        x_keys = torch.tensor(
            [item[0] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.long
        )
        x_timestamps = torch.tensor(
            [item[1] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        )

        # the next value in the sequence (y)
        y_key = torch.tensor(
            self.data[idx + self.seq_len][0],
            dtype=torch.long
        )

        return (x_keys, x_timestamps), y_key