import torch
import torch.nn as nn
from utils.log_utils import debug, info


class LSTM(nn.Module):

    def _set_fields(self, params, config_settings):
        """
        Method to set the model's parameters.
        :param params: The model's parameters.
        :param config_settings: The configuration settings.
        :return:
        """
        # initial message
        info("🔄 LSTM fields setting started...")

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
                # check if it is passed to the model
                if param in params and params[param] is not None:
                    debug(f"⚙️ '{param}' found. Using the specified value ({params[param]}).")
                    # set the parameter to the specified value
                    setattr(self, param, params[param])
                else:
                    # otherwise, take the parameter from configuration settings
                    config_value = getattr(config_settings, param)
                    debug(f"⚙️ '{param}' not found or is None. Using config value ({config_value}).")
                    # and set the parameter to that value
                    setattr(self, param, config_value)

            # apply dropout only if num_layers > 1
            if self.num_layers > 1:
                self.dropout = float(self.dropout)
                debug(f"⚙️ 'dropout' applied: {self.dropout}")
            else:
                debug(f"⚙️ 'dropout' cannot be applied with num_layers = {self.num_layers}.")
                self.dropout = 0.0

            # set the no. of keys
            self.num_keys = config_settings.num_keys

            # apply embedding
            self.embedding_dim = config_settings.embedding_dim

            # instantiate the embedding layer
            self.embedding = nn.Embedding(
                self.num_keys,
                self.embedding_dim
            )

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise RuntimeError(f"❌ Error while setting class fields: {e}.")

        # show a successful message
        info("🟢 LSTM fields set.")


    def __init__(self, params, config_settings):
        """
        Method to initialize the LSTM model.
        :param params: The hyperparameters of the model.
        :param config_settings: The configuration settings.
        """
        # initial message
        info("🔄 LSTM initialization started...")

        super(LSTM, self).__init__()

        self.use_mc_dropout = False

        # set model's parameters
        self._set_fields(params, config_settings)

        # debugging
        for param in self.required_parameters:
            debug(f"⚙️ {param} = {getattr(self, param)}.")

        try:
            # instantiate the LSTM model
            self.lstm = nn.LSTM(
                input_size=config_settings.num_features
                +self.embedding_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=self.batch_first,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                proj_size=self.proj_size
            )
        except (TypeError, ValueError, KeyError) as e:
            raise RuntimeError(f"❌ Error while instantiating LSTM model: {e}.")

        try:
            self.mc_dropout_layer = nn.Dropout(p=self.dropout)
        except (TypeError, AttributeError) as e:
            raise RuntimeError(f"❌ Error while instantiating the Dropout layer: {e}.")
        try:
            # fully-connected layer (linear)
            self.fc = nn.Linear(self.hidden_size, self.num_keys)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"❌ Error while instantiating the FC layer: {e}.")

        # show a successful message
        info("🟢 LSTM initialized.")


    def _get_lstm_input(self, x_features, x_keys):
        """
        Method to prepare and return the LSTM input.
        :param x_features: The features to pass to the model.
        :param x_keys: The keys to embed and pass to the model.
        :return: The LSTM input.
        """
        # initial message
        info("🔄 LSTM input retrieval started...")

        # check the inputs validity
        if x_features is None or x_keys is None:
            raise ValueError("❌ Input features cannot be None.")

        # debugging
        debug(f"⚙️ Input features shape: {x_features.shape}.")
        debug(f"⚙️ Input keys shape: {x_keys.shape}.")

        # pass the key through the embedding layer
        embedded_keys = self.embedding(x_keys)

        # debugging
        debug(f"⚙️ Embedded keys shape: {embedded_keys.shape}.")

        try:
            # concatenate embedding layer with the other features
            x = torch.cat(
                (x_features, embedded_keys),
                dim=-1
            )
        except (RuntimeError, TypeError) as e:
            raise RuntimeError(f"❌ Error while constructing the LSTM input: {e}.")

        # show a successful message
        info("🟢 LSTM input retrieved.")

        return x


    def forward(self, x_features, x_keys):
        """
        Method to perform the forward pass through the LSTM.
        :param x_features: The features.
        :param x_keys: The keys.
        :return: The logits of the LSTM.
        """
        # initial message
        info("🔄 LSTM forward started...")

        try:
            # get the input of the LSTM
            x = self._get_lstm_input(x_features, x_keys)

            # pass the features to the LSTM
            lstm_out, _ = self.lstm(x)

            # check if use dropout
            if self.use_mc_dropout:
                lstm_out = self.mc_dropout_layer(lstm_out)

            # pass through the fully-connected layer
            logits = self.fc(lstm_out[:, -1, :])

        except (AttributeError, TypeError, ValueError) as e:
            raise RuntimeError(f"❌ Error while passing data through LSTM: {e}.")

        # debugging
        debug(f"⚙️ Output shape: {lstm_out.shape}.")

        try:
            # get the logits from the LSTM output
            logits = self.fc(lstm_out[:, -1, :])
        except (IndexError, AttributeError) as e:
            raise RuntimeError(f"❌ Error while processing LSTM output: {e}.")

        # debugging
        debug(f"⚙️ Logits shape: {logits.shape}.")

        # show a successful message
        info("🟢 LSTM forward completed.")

        return logits