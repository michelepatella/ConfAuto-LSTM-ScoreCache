import torch
import torch.nn as nn
from utils.log_utils import debug, info
from utils.config_utils import get_config_value


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
                # check if the parameter has been passed
                if param in params:
                    # apply all the other parameters (except dropout), if specified
                    if (params[param] is not None and
                            params[param] != "dropout"):

                        # debugging
                        debug(f"⚙️ '{param}' found. Using the specified "
                              f"value ({params[param]}).")

                        # set the value
                        setattr(self, param, params[param])
                    else:
                        # debugging
                        debug(f"⚙️ '{param}' not found. Using the default value.")

                        # if they are None, read them from config file and set them
                        setattr(
                            self,
                            param,
                            get_config_value(
                                config_settings.config,
                                f"model.params.{param}"
                            )
                        )
                else:
                    # debugging
                    debug(f"⚙️ '{param}' not found. Using the default value.")

                    # read the required parameter from config
                    setattr(
                        self,
                        param,
                        get_config_value(
                            config_settings.config,
                            f"model.params.{param}"
                            )
                    )

            # check if dropout can be applied
            if params.get(
                    "num_layers",
                    config_settings.num_layers
            ) > 1:

                # debugging
                debug(f"⚙️ 'dropout' can be applied.")

                # apply dropout
                if params["dropout"] is not None:
                    # debugging
                    debug(f"⚙️ 'dropout' found. Using the specified "
                          f"value ({float(params['dropout'])}).")

                    # set the value
                    setattr(self, "dropout", float(params["dropout"]))
                else:
                    # debugging
                    debug(f"⚙️ 'dropout' not found. Using the default value.")

                    # set the value
                    setattr(self, "dropout", float(config_settings.dropout))
            else:
                # debugging
                debug(f"⚙️ 'dropout' cannot be applied.")

                # dropout cannot be applied
                setattr(self, "dropout", 0.0)

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