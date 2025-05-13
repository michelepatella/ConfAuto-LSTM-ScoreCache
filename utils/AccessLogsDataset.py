import torch
from torch.utils.data import Dataset
from utils.config_utils import _get_config_value
from utils.dataset_utils import _load_dataset


class AccessLogsDataset(Dataset):

    def _split_dataset(self, dataset_type):
        """
        Method to split the dataset based on the requested type.
        :param dataset_type: The dataset type requested ("training" or "testing").
        :return:
        """
        try:
            # define the splitting's index
            split_idx = int(
                len(self.data) * _get_config_value("data.training_perc")
            )
        except Exception as e:
            raise Exception(f"❌ Error while defining the dataset splitting's index: {e}")

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


    def _set_fields(self, df):
        """
        Method to set the fields of the dataset.
        :param df: The dataframe from which the fields are extracted.
        :return:
        """
        try:
            # assuming all the columns are features
            # while the last one is the target to predict
            self.columns = df.columns.tolist()
            self.features = self.columns[:-1]
            self.target = self.columns[-1]

            # set all the fields dynamically
            for column in df.columns:
                setattr(self, column, df[column].values)

        except Exception as e:
            raise Exception(f"❌ Error while reading the dataset columns: {e}")

        # set the sequence length
        self.seq_len = _get_config_value("data.seq_len")

        # set data equals created fields
        self.data = list(zip(*[
            getattr(self, column)
            for column in self.columns
        ]))


    def __init__(self, dataset_path, dataset_type):
        """
        Method to instantiate the access logs dataset.
        :param dataset_path: The path of the access logs dataset.
        :param dataset_type: The type of dataset requested ("training" or "testing").
        """
        # load the dataset
        df = _load_dataset(dataset_path)

        # set the fields of the dataset
        self._set_fields(df)

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
        try:
            x_features_list = []

            # for all the features of the dataset
            for feature in self.features:

                # get the feature in the sequence
                feature_data = torch.tensor(
                    [getattr(self, feature)[i] for i
                     in range(idx, idx + self.seq_len)],
                    dtype=torch.float
                ).unsqueeze(-1)

                # add the feature to the features list
                x_features_list.append(feature_data)

        except Exception as e:
            raise Exception(f"❌ Error while reading the sequence (x): {e}")

        try:
            # combine all features
            x_features = torch.cat(x_features_list, dim=-1)
        except Exception as e:
            raise Exception(f"❌ Error while combining features: {e}")

        return x_features


    def _get_next_seq_value(self, idx):
        """
        Method to get the next sequence value from the access logs dataset.
        :param idx: Index of the access logs dataset.
        :return:
        """
        try:
            # the next value in the sequence (y)
            y_key = torch.tensor(
                getattr(self, self.target)[idx + self.seq_len],
                dtype=torch.long
            )
        except Exception as e:
            raise Exception(f"❌ Error while reading the next value in the sequence (y): {e}")

        return y_key


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

        # get the next value in the sequence
        y_key = self._get_next_seq_value(idx)

        return x_features, y_key