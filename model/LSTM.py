import torch.nn as nn
from utils.config_utils import _get_config_value


class LSTM(nn.Module):

    def _set_fields(self, params):
        """
        Method to set the model's parameters.
        :param params: The model's parameters.
        :return:
        """
        # define the model's config (+ num_keys)
        self.hidden_size = params["hidden_size"] \
            if params["hidden_size"] is not None \
            else _get_config_value("model.hidden_size")

        self.num_layers = params["num_layers"] \
            if params["num_layers"] is not None \
            else _get_config_value("model.num_layers")

        self.bias = params["bias"] \
            if params["bias"] is not None \
            else _get_config_value("model.bias")

        self.batch_first = params["batch_first"] \
            if params["batch_first"] is not None \
            else _get_config_value("model.batch_first")

        if params["dropout"] is not None:
            self.dropout = float(params["dropout"])
        elif self.num_layers > 1:
            self.dropout = float(_get_config_value("model.dropout"))
        else:
            self.dropout = 0.0

        self.bidirectional = params["bidirectional"] \
            if params["bidirectional"] is not None \
            else _get_config_value("model.bidirectional")

        self.proj_size = params["proj_size"] \
            if params["proj_size"] is not None \
            else _get_config_value("model.proj_size")

        self.num_keys = params["num_keys"] \
            if params["num_keys"] is not None \
            else _get_config_value("data.num_keys")


    def __init__(self, params):
        """
        Method to initialize the LSTM model.
        :param params: The hyperparameters of the model.
        """
        super(LSTM, self).__init__()

        # set model's parameters
        self._set_fields(params)

        try:
            # instantiate the LSTM model
            self.lstm = nn.LSTM(
                input_size=_get_config_value("model.features_dim")
                           +self.num_keys,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=self.batch_first,
                dropout=self.dropout,
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