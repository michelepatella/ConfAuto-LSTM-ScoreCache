import torch.nn as nn
from utils.config_utils import _load_config, _get_config_value


class LSTM(nn.Module):

    def __init__(
            self,
            hidden_size=None,
            num_layers=None,
            bias=None,
            batch_first=None,
            dropout=None,
            bidirectional=None,
            proj_size=None,
            num_keys=None,
    ):
        """
        Method to initialize the LSTM model.
        :param hidden_size: Number of hidden features.
        :param num_layers: Number of LSTM layers.
        :param bias: Indicates whether bias should be applied.
        :param batch_first: Indicates whether batch_first should be applied.
        :param dropout: Dropout layer to apply.
        :param bidirectional: Indicates if the LSTM should be bidirectional.
        :param proj_size: Size of the projection layer.
        :param num_keys: Number of cache keys.
        """
        super(LSTM, self).__init__()

        # load model and data configs
        config = _load_config()

        # define the model's config (+ num_keys)
        self.hidden_size = hidden_size \
            if hidden_size is not None \
            else _get_config_value("model.hidden_size")
        self.num_layers = num_layers \
            if num_layers is not None \
            else _get_config_value("model.num_layers")
        self.bias = bias \
            if bias is not None \
            else _get_config_value("model.bias")
        self.batch_first = batch_first \
            if batch_first is not None \
            else _get_config_value("model.batch_first")
        self.dropout = float(dropout) \
            if dropout is not None \
            else float(_get_config_value("model.dropout"))
        self.bidirectional = bidirectional \
            if bidirectional is not None \
            else _get_config_value("model.bidirectional")
        self.proj_size = proj_size \
            if proj_size is not None \
            else _get_config_value("model.proj_size")
        self.num_keys = num_keys \
            if num_keys is not None \
            else _get_config_value("data.num_keys")

        # check if dropout should be applied or not
        effective_dropout = self.dropout if self.num_layers > 1 else 0.0

        try:
            # instantiate the LSTM model
            self.lstm = nn.LSTM(
                input_size=5,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=self.batch_first,
                dropout=effective_dropout,
                bidirectional=self.bidirectional,
                proj_size=self.proj_size
            )
        except Exception as e:
            raise Exception(f"❌ Error while instantiating LSTM model: {e}")

        try:
            # fully-connected layer (linear)
            self.fc = nn.Linear(self.hidden_size, self.num_keys + 1)
        except Exception as e:
            raise Exception(f"❌ Error while instantiating the FC layer: {e}")


    def forward(self, x_features):
        """
        Method to perform the forward pass through the LSTM.
        :param x_features: The features.
        :return: The logits of the LSTM.
        """
        # check the inputs validity
        if x_features is None:
            raise ValueError("❌ Input features cannot be None.")

        try:
            # pass the features to the LSTM
            lstm_out, _ = self.lstm(x_features)
        except Exception as e:
            raise Exception(f"❌ Error while passing data through LSTM: {e}")

        try:
            # get the logits from the LSTM output
            logits = self.fc(lstm_out[:, -1, :])
        except Exception as e:
            raise Exception(f"❌ Error while processing LSTM output: {e}")

        return logits