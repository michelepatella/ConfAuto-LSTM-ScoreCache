import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.config_loader import load_config

class AccessLogsDataset(Dataset):

    def __init__(self, csv_path):
        """
        Method to instantiate the access logs dataset.
        :param csv_path: The csv path of the access logs dataset.
        """
        # load data configuration
        config = load_config()
        data_config = config['data']

        # get the csv dataset file
        df = pd.read_csv(csv_path)

        # set keys and seq_len fields
        self.keys = df['key'].values
        self.seq_len = data_config['time_steps']

    def __len__(self):
        """
        Method to return the length of the access logs dataset.
        :return: The length of the access logs dataset as output.
        """
        return len(self.keys) - self.seq_len

    def __getitem__(self, idx):
        """
        Method to return a sequence and the next value in the sequence.
        :param idx: The index of the access logs dataset
        :return: Access logs dataset as output.
        """
        # get the sequence (x)
        x = torch.tensor(self.keys[idx:idx + self.seq_len], dtype=torch.long)

        # the next value in the sequence (y)
        y = torch.tensor(self.keys[idx + self.seq_len], dtype=torch.long)

        return x, y

