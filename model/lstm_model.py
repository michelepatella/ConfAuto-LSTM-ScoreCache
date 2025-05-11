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
            num_keys=None,
            use_embedding=True
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
        :param num_keys: Number of cache keys.
        :param use_embedding: Speifies whether to use embeddings.
        """
        super(LSTM, self).__init__()

        # load model and data configs
        config = load_config()
        model_config = config["model"]
        data_config = config["data"]

        # define the model's config (+ num_keys)
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
        self.dropout = float(dropout) \
            if dropout is not None \
            else float(model_config["dropout"])
        self.bidirectional = bidirectional \
            if bidirectional is not None \
            else model_config["bidirectional"]
        self.proj_size = proj_size \
            if proj_size is not None \
            else model_config["proj_size"]
        self.num_keys = num_keys \
            if num_keys is not None \
            else data_config["num_keys"]
        self.use_embedding = use_embedding

        if self.use_embedding:
            # embedding layer for keys
            self.embedding = nn.Embedding(
                num_embeddings=self.num_keys + 1,
                embedding_dim=self.embedding_dim
            )
            input_size = 5 + self.embedding_dim
        else:
            input_size = 5

        # check if dropout should be applied or not
        effective_dropout = self.dropout if self.num_layers > 1 else 0.0

        try:
            # instantiate the LSTM model
            self.lstm = nn.LSTM(
                input_size=input_size,  # features + embedding
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=self.batch_first,
                dropout=effective_dropout,
                bidirectional=self.bidirectional,
                proj_size=self.proj_size
            )
        except:
            raise Exception("An unexpected error while instantiating LSTM model.")

        # fully-connected layer (linear)
        self.fc = nn.Linear(self.hidden_size, self.num_keys + 1)

    def forward(self, x_features, keys=None):
        """
        Method to perform the forward pass through the LSTM.
        :param x_features: The 4 features (hour of day sin, hour of day cos,
        day of week sin, day of week cos).
        :param keys: The keys requested.
        :return: The logits of the LSTM as output
        """
        # check the inputs validity
        if x_features is None:
            raise ValueError("Input features cannot be None.")

        try:
            # embedding used only while training
            if self.use_embedding:
                if keys is not None:
                    key_embeddings = self.embedding(keys)
                    # concatenate features and keys
                    x_cat = torch.cat([x_features, key_embeddings], dim=-1)
                else:
                    raise ValueError("Keys not passed to the model.")
            else:
                # embedding not used at prediction time
                x_cat = x_features
        except Exception as e:
            raise Exception(f"An unexpected error while processing inputs: {e}")

        try:
            # pass the features to the LSTM
            lstm_out, _ = self.lstm(x_cat)
        except Exception as e:
            raise Exception(f"An unexpected error while passing data through LSTM: {e}")

        try:
            # get the logits from the LSTM output
            logits = self.fc(lstm_out[:, -1, :])
        except Exception as e:
            raise Exception(f"An unexpected error while processing LSTM output: {e}")

        return logits