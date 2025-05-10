import torch
import torch.nn as nn
from utils.config_loader import load_config

class LSTM(nn.Module):

    def __init__(
            self,
            embedding_dim=None,
            hidden_size=None,
            num_layers=None,
            bias=None,
            batch_first=None,
            dropout=None,
            bidirectional=None,
            proj_size=None,
            n_keys=None
    ):
        """
        Method to initialize the LSTM model.
        :param embedding_dim: Embedding layer dimension.
        :param hidden_size: Number of hidden features.
        :param num_layers: Number of LSTM layers.
        :param bias: Indicates whether bias should be applied.
        :param batch_first: Indicates whether batch_first should be applied.
        :param dropout: Dropout layer to apply.
        :param bidirectional: Indicates if the LSTM should be bidirectional.
        :param proj_size: Size of the projection layer.
        :param n_keys: Number of cache keys.
        """
        super(LSTM, self).__init__()

        # load model and data configurations
        config = load_config()
        model_config = config["model"]
        data_config = config["data"]

        # define the model's configuration (+ n_keys)
        self.embedding_dim = embedding_dim \
            if embedding_dim is not None \
            else model_config["embedding_dim"]
        self.hidden_size = hidden_size \
            if hidden_size is not None \
            else model_config["hidden_size"]
        self.num_layers = num_layers \
            if num_layers is not None \
            else model_config["num_layers"]
        self.bias = bias \
            if bias is not None \
            else model_config["bias"]
        self.batch_first = batch_first \
            if batch_first is not None \
            else model_config["batch_first"]
        self.dropout = dropout \
            if dropout is not None \
            else model_config["dropout"]
        self.bidirectional = bidirectional \
            if bidirectional is not None \
            else model_config["bidirectional"]
        self.proj_size = proj_size \
            if proj_size is not None \
            else model_config["proj_size"]
        self.n_keys = n_keys \
            if n_keys is not None \
            else data_config["n_keys"]

        # embedding layer for keys
        self.embedding = nn.Embedding(
            num_embeddings=self.n_keys + 1,
            embedding_dim=self.embedding_dim
        )

        # instantiate the LSTM model
        self.lstm = nn.LSTM(
            6, # features + timestamp
            self.embedding_dim + 1,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.dropout,
            self.bidirectional,
            self.proj_size
        )

        # fully-connected layer (linear)
        self.fc = nn.Linear(self.hidden_size, self.n_keys + 1)

    def forward(self, x_features, x_timestamps, keys):
        """
        Method to perform the forward pass through the LSTM.
        :param x_features: The 4 features (hour of day sin, hour of day cos,
        day of week sin, day of week cos).
        :param x_timestamps: The timestamps of the 4 features.
        :param keys: The keys requested.
        :return: The logits of the LSTM as output
        """
        # concatenate features and timestamps
        timestamps = x_timestamps.unsqueeze(-1)

        # get embedding for the keys
        key_embeddings = self.embedding(keys)

        # concatenate features and keys
        x_cat = torch.cat([x_features, timestamps, key_embeddings], dim=-1)

        # pass the features to the LSTM
        lstm_out, _ = self.lstm(x_cat)

        # get the logits
        logits = self.fc(lstm_out[:, -1, :])

        return logits