import torch
import torch.nn as nn
import logging
from utils.config_utils import _get_config_value


class LSTM(nn.Module):

    def _set_fields(self, params):
        """
        Method to set the model's parameters.
        :param params: The model's parameters.
        :return:
        """
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
            self.num_keys = _get_config_value(f"data.num_keys")

            # embed the keys
            self.key_embedding = nn.Embedding(
                self.num_keys,
                embedding_dim=_get_config_value("model.embedding_dim")
            )

        except Exception as e:
            raise Exception(f"❌ Error while setting LSTM fields: {e}")


    def __init__(self, params):
        """
        Method to initialize the LSTM model.
        :param params: The hyperparameters of the model.
        """
        super(LSTM, self).__init__()

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

        # set model's parameters
        self._set_fields(params)

        # debugging
        for param in self.required_parameters:
            logging.debug(f"⚙️ {param} = {getattr(self, param)}.")

        try:
            # instantiate the LSTM model
            self.lstm = nn.LSTM(
                input_size=1+_get_config_value("model.embedding_dim"),
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


    def forward(self, x_features):
        """
        Method to perform the forward pass through the LSTM.
        :param x_features: The features.
        :return: The logits of the LSTM.
        """
        # check the inputs validity
        if x_features is None:
            raise ValueError("❌ Input features cannot be None.")

        # debugging
        logging.debug(f"⚙️ Input shape: {x_features.shape}.")

        # composition of the input for LSTM
        delta_time = x_features[:, :, 0].unsqueeze(-1)
        key_id = x_features[:, :, 1].long()
        key_emb = self.key_embedding(key_id)
        lstm_input = torch.cat(
            [delta_time, key_emb],
            dim=-1
        )

        try:
            # pass the features to the LSTM
            lstm_out, _ = self.lstm(lstm_input)
        except Exception as e:
            raise Exception(f"❌ Error while passing data through LSTM: {e}")

        # debugging
        logging.debug(f"⚙️ Output shape: {lstm_out.shape}.")

        try:
            # get the logits from the LSTM output
            logits = self.fc(lstm_out[:, -1, :])
        except Exception as e:
            raise Exception(f"❌ Error while processing LSTM output: {e}")

        # debugging
        logging.debug(f"⚙️ Logits shape: {logits.shape}.")

        return logits