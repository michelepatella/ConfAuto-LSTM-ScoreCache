import torch
from torch.utils.data import Dataset
from utils.config_utils import _get_config_value
from utils.dataset_utils import _load_dataset
from sklearn.preprocessing import OneHotEncoder


class AccessLogsDataset(Dataset):

    def _split_dataset(self, dataset_type):
        """
        Method to split the dataset based on the requested type.
        :param dataset_type: The dataset type requested ("training" or "testing").
        :return:
        """
        try:
            # define the splitting's index
            split_idx = int(len(self.requests) *
                            _get_config_value("data.training_perc"))
        except Exception as e:
            raise Exception(f"❌ Error while defining the dataset splitting's index: {e}")

        try:
            # split the dataset
            if dataset_type == "training":
                self.targets = self.requests_int[:split_idx]
                self.data = list(
                    zip(
                        self.requests[:split_idx],
                        self.timestamps[:split_idx],
                        self.hour_of_day_cos[:split_idx],
                        self.hour_of_day_sin[:split_idx],
                        self.day_of_week_cos[:split_idx],
                        self.day_of_week_sin[:split_idx]
                    )
                )
            elif dataset_type == "testing":
                self.targets = self.requests_int[split_idx:]
                self.data = list(
                    zip(
                        self.requests[split_idx:],
                        self.timestamps[split_idx:],
                        self.hour_of_day_cos[split_idx:],
                        self.hour_of_day_sin[split_idx:],
                        self.day_of_week_cos[split_idx:],
                        self.day_of_week_sin[split_idx:]
                    )
                )
            else:
                raise ValueError("❌ Invalid split type.")
        except Exception as e:
            raise Exception(f"❌ Error while splitting the dataset: {e}")


    def __init__(self, dataset_path, dataset_type):
        """
        Method to instantiate the access logs dataset.
        :param dataset_path: The path of the access logs dataset.
        :param dataset_type: The type of dataset requested ("training" or "testing").
        """
        # load the dataset
        df = _load_dataset(dataset_path)

        try:
            # set all the fields
            self.timestamps = df["timestamp"].values
            self.hour_of_day_sin = df["hour_of_day_sin"].values
            self.hour_of_day_cos = df["hour_of_day_cos"].values
            self.day_of_week_sin = df["day_of_week_sin"].values
            self.day_of_week_cos = df["day_of_week_cos"].values
            self.requests_int = df["request"].values
            self.seq_len = _get_config_value("data.seq_len")
        except Exception as e:
            raise Exception(f"❌ Error while reading the dataset fields: {e}")

        # instantiate the encoder
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.requests = self.encoder.fit_transform(df[["request"]])

        # split the dataset to assign data properly
        self._split_dataset(dataset_type)


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
        try:
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

            x_requests = torch.tensor(
                [item[0] for item in self.data[idx:idx + self.seq_len]],
                dtype=torch.float
            )
        except Exception as e:
            raise Exception(f"❌ Error while reading the sequence (x): {e}")

        try:
            # combine all features
            x_features = torch.cat([
                x_timestamps,
                x_hour_of_day_cos,
                x_hour_of_day_sin,
                x_day_of_week_cos,
                x_day_of_week_sin,
                x_requests
            ], dim=-1)
        except Exception as e:
            raise Exception(f"❌ Error while combining features: {e}")

        return x_features


    def __getitem__(self, idx):
        """
        Method to return a sequence and the next value in the sequence.
        :param idx: The index of the access logs dataset.
        :return: Access logs dataset.
        """
        # check idx + seq_len does not exceed bounds
        if idx + self.seq_len >= len(self.data):
            raise IndexError(f"❌ Index {idx} out of bounds for sequence length {self.seq_len}.")

        # get the features
        x_features = self._get_features(idx)

        try:
            # the next value in the sequence (y)
            y_key = torch.tensor(
                self.targets[idx + self.seq_len],
                dtype=torch.long
            )
        except Exception as e:
            raise Exception(f"❌ Error while reading the next value in the sequence (y): {e}")

        return x_features, y_key