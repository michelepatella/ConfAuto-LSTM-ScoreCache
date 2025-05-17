import torch
import torch.nn as nn
from utils.log_utils import _debug
from utils.config_utils import _get_config_value


class LSTM(nn.Module):

    def _set_fields(self, params):
        """
        Method to set the model's parameters.
        :param params: The model's parameters.
        :return:
        """

        # definition of the required parameters
        self.required_parameters = [
            "hidden_size",
            "num_layers",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
            "proj_size"
        ]

        try:
            # for each required parameter
            for param in self.required_parameters:
                # check if the parameter has been passed
                if param in params:
                    # apply all the other parameters (except dropout), if specified
                    if (params[param] is not None and
                            params[param] != "dropout"):
                        setattr(self, param, params[param])
                    else:
                        # if they are None, read them from config file and set them
                        setattr(
                            self,
                            param,
                            _get_config_value(f"model.params.{param}")
                        )
                else:
                    # read the required parameter from config
                    setattr(
                        self,
                        param,
                        _get_config_value(f"model.params.{param}")
                    )

            # check if dropout can be applied
            if params.get(
                    "num_layers",
                    _get_config_value("model.params.num_layers")
            ) > 1:
                # apply dropout
                if params["dropout"] is not None:
                    setattr(self, "dropout", float(params["dropout"]))
                else:
                    setattr(
                        self,
                        "dropout",
                        float(_get_config_value("model.params.dropout"))
                    )
            else:
                # dropout cannot be applied
                setattr(self, "dropout", 0.0)

            # set the no. of keys
            self.num_keys = _get_config_value("data.num_keys")

            # apply embedding
            self.embedding_dim = _get_config_value("data.embedding_dim")

            # instantiate the embedding layer
            self.embedding = nn.Embedding(
                self.num_keys,
                self.embedding_dim
            )

        except Exception as e:
            raise Exception(f"❌ Error while setting LSTM fields: {e}")


    def __init__(self, params):
        """
        Method to initialize the LSTM model.
        :param params: The hyperparameters of the model.
        """
        super(LSTM, self).__init__()

        # set model's parameters
        self._set_fields(params)

        # debugging
        for param in self.required_parameters:
            _debug(f"⚙️ {param} = {getattr(self, param)}.")

        try:
            # instantiate the LSTM model
            self.lstm = nn.LSTM(
                input_size=_get_config_value("model.num_features")
                           + self.embedding_dim,
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
            self.fc = nn.Linear(self.hidden_size, self.num_keys)
        except Exception as e:
            raise Exception(f"❌ Error while instantiating the FC layer: {e}")


    def _get_lstm_input(self, x_features, x_keys):
        """
        Method to prepare and return the LSTM input.
        :param x_features: The features to pass to the model.
        :param x_keys: The keys to embed and pass to the model.
        :return: The LSTM input.
        """
        # check the inputs validity
        if x_features is None or x_keys is None:
            raise ValueError("❌ Input features cannot be None.")

        # debugging
        _debug(f"⚙️ Input features shape: {x_features.shape}.")
        _debug(f"⚙️ Input keys shape: {x_keys.shape}.")

        # pass the key through the embedding layer
        embedded_keys = self.embedding(x_keys)
        _debug(f"⚙️ Embedded keys shape: {embedded_keys.shape}.")

        # concatenate embedding layer with the other features
        x = torch.cat(
            (x_features, embedded_keys),
            dim=-1
        )

        return x


    def forward(self, x_features, x_keys):
        """
        Method to perform the forward pass through the LSTM.
        :param x_features: The features.
        :param x_keys: The keys.
        :return: The logits of the LSTM.
        """
        try:
            # get the input of the LSTM
            x = self._get_lstm_input(x_features, x_keys)

            # pass the features to the LSTM
            lstm_out, _ = self.lstm(x)
        except Exception as e:
            raise Exception(f"❌ Error while passing data through LSTM: {e}")

        # debugging
        _debug(f"⚙️ Output shape: {lstm_out.shape}.")

        try:
            # get the logits from the LSTM output
            logits = self.fc(lstm_out[:, -1, :])
        except Exception as e:
            raise Exception(f"❌ Error while processing LSTM output: {e}")

        # debugging
        _debug(f"⚙️ Logits shape: {logits.shape}.")

        return logits