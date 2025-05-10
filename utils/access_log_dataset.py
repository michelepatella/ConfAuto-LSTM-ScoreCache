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

        # set all the fields
        self.timestamps = df["timestamp"].values
        self.hour_of_day_sin = df["hour_of_day_sin"].values
        self.hour_of_day_cos = df["hour_of_day_cos"].values
        self.day_of_week_sin = df["day_of_week_sin"].values
        self.day_of_week_cos = df["day_of_week_cos"].values
        self.keys = df["key"].values

        # set the sequence length
        self.seq_len = data_config["seq_len"]

        # split dataset into training, testing, and validation set
        split_idx_1 = int(len(self.keys) * data_config["training_perc"])
        split_idx_2 = int(len(self.keys) *
                          (data_config["training_perc"] + data_config["validation_perc"]))

        # split the dataset
        if split == "training":
            self.data = list(
                zip(
                    self.keys[:split_idx_1],
                    self.timestamps[:split_idx_1],
                    self.hour_of_day_cos[:split_idx_1],
                    self.hour_of_day_sin[:split_idx_1],
                    self.day_of_week_cos[:split_idx_1],
                    self.day_of_week_sin[:split_idx_1]
                )
            )
        elif split == "validation":
            self.data = list(
                zip(
                    self.keys[split_idx_1:split_idx_2],
                    self.timestamps[split_idx_1:split_idx_2],
                    self.hour_of_day_cos[split_idx_1:split_idx_2],
                    self.hour_of_day_sin[split_idx_1:split_idx_2],
                    self.day_of_week_cos[split_idx_1:split_idx_2],
                    self.day_of_week_sin[split_idx_1:split_idx_2]
                )
            )
        elif split == "testing":
            self.data = list(
                zip(
                    self.keys[split_idx_2:],
                    self.timestamps[split_idx_2:],
                    self.hour_of_day_cos[split_idx_2:],
                    self.hour_of_day_sin[split_idx_2:],
                    self.day_of_week_cos[split_idx_2:],
                    self.day_of_week_sin[split_idx_2:]
                )
            )
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
        x_hour_of_day_cos = torch.tensor(
            [item[2] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        )
        x_hour_of_day_sin = torch.tensor(
            [item[3] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        )
        x_day_of_week_cos = torch.tensor(
            [item[4] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        )
        x_day_of_week_sin = torch.tensor(
            [item[5] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        )

        # combine all features
        x_features = torch.stack(
            [x_hour_of_day_cos, x_hour_of_day_sin, x_day_of_week_cos, x_day_of_week_sin],
            dim=-1
        )

        # the next value in the sequence (y)
        y_key = torch.tensor(
            self.data[idx + self.seq_len][0],
            dtype=torch.long
        )

        return (x_keys, x_timestamps, x_features), y_key