import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.config_utils import load_config, get_config_value


class AccessLogsDataset(Dataset):

    def _split_dataset(self, split, training_perc, validation_perc):
        """
        Method to split the dataset.
        :param split: Type of split.
        :param training_perc: Percentage of training set.
        :param validation_perc: Percentage of validation set.
        :return:
        """
        try:
            # define the splittings
            split_idx_1 = int(len(self.keys) * training_perc)
            split_idx_2 = int(len(self.keys) * (training_perc + validation_perc))
        except Exception as e:
            raise Exception(f"An unexpected error while defining the splittings: {e}")

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

    def __init__(self, csv_path, split):
        """
        Method to instantiate the access logs dataset.
        :param csv_path: The csv path of the access logs dataset.
        :param split: The type of dataset split ("training", "validation", or "testing").
        """
        # load config file
        config = load_config()

        # get the csv dataset file
        df = pd.read_csv(csv_path)

        # try to set the fields
        try:
            # set all the fields
            self.timestamps = df["timestamp"].values
            self.hour_of_day_sin = df["hour_of_day_sin"].values
            self.hour_of_day_cos = df["hour_of_day_cos"].values
            self.day_of_week_sin = df["day_of_week_sin"].values
            self.day_of_week_cos = df["day_of_week_cos"].values
            self.keys = df["key"].values
            self.seq_len = get_config_value(config, "data.seq_len")
        except Exception as e:
            raise Exception(f"An unexpected error while reading the access logs dataset: {e}")


        # split the dataset
        self._split_dataset(
            split,
            get_config_value(config, "data.training_perc"),
            get_config_value(config, "data.validation_perc")
        )

    def __len__(self):
        """
        Method to return the length of the access logs dataset.
        :return: The length of the access logs dataset.
        """
        return len(self.data) - self.seq_len

    def _get_features(self, idx):
        """
        Method to get the features from the access logs dataset.
        :param idx: Index of the access logs dataset.
        :return: The features of the access logs dataset.
        """
        # get the sequence (x)
        x_timestamps = torch.tensor(
            [item[1] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        ).unsqueeze(-1)

        x_hour_of_day_cos = torch.tensor(
            [item[2] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        ).unsqueeze(-1)

        x_hour_of_day_sin = torch.tensor(
            [item[3] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        ).unsqueeze(-1)

        x_day_of_week_cos = torch.tensor(
            [item[4] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        ).unsqueeze(-1)

        x_day_of_week_sin = torch.tensor(
            [item[5] for item in self.data[idx:idx + self.seq_len]],
            dtype=torch.float
        ).unsqueeze(-1)

        # combine all features
        x_features = torch.cat([
            x_timestamps,
            x_hour_of_day_cos,
            x_hour_of_day_sin,
            x_day_of_week_cos,
            x_day_of_week_sin],
            dim=-1
        )

        return x_features

    def __getitem__(self, idx):
        """
        Method to return a sequence and the next value in the sequence.
        :param idx: The index of the access logs dataset.
        :return: Access logs dataset.
        """
        # check idx + seq_len does not exceed bounds
        if idx + self.seq_len >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for sequence length {self.seq_len}.")

        # get the features
        x_features = self._get_features(idx)

        # the next value in the sequence (y)
        y_key = torch.tensor(
            self.data[idx + self.seq_len][0],
            dtype=torch.long
        )

        return x_features, y_key