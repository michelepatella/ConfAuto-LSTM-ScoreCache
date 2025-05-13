import torch.nn as nn
from utils.config_utils import _get_config_value


class LSTM(nn.Module):

    def _set_fields(self, params):
        """
        Method to set the model's parameters.
        :param params: The model's parameters.
        :return:
        """
        # define the model's parameters (+ num_keys)
        for key, value in params.items():

            # handle dropout separately
            if key == "dropout":

                # check if dropout can be applied
                if params.get(
                        "num_layers",
                        _get_config_value("model.num_layers")
                ) > 1:
                    # apply dropout
                    if value is not None:
                        setattr(self, key, float(value))
                    else:
                        setattr(self, key, float(_get_config_value(f"model.{key}")))
                else:
                    # dropout cannot be applied
                    setattr(self, key, 0.0)

            else:

                # apply all the other parameters, if specified
                if value is not None:
                    setattr(self, key, value)
                else:
                    # if they are None, read them from config file and set them
                    config_section = "data" \
                        if key == "num_keys" \
                        else "model"
                    setattr(self, key, _get_config_value(f"{config_section}.{key}"))


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
                           + self.num_keys,
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