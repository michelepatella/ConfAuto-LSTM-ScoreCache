import torch.nn as nn
from utils.config_loader import load_config

class LSTM(nn.Module):

    def __init__(
            self,
            input_size=None,
            hidden_size=None,
            num_layers=None,
            bias=None,
            batch_first=None,
            dropout=None,
            bidirectional=None,
            proj_size=None,
            num_keys=None
    ):
        """
        Method to initialize the LSTM model with output probabilities.
        :param input_size: Number of input features
        :param hidden_size: Number of hidden features
        :param num_layers: Number of LSTM layers
        :param bias: Indicates whether bias should be applied
        :param batch_first: Indicates whether batch_first should be applied
        :param dropout: Dropout layer to apply
        :param bidirectional: Indicates if the LSTM should be bidirectional
        :param proj_size: Size of the projection layer
        :param num_keys: Number of cache keys
        """
        super(LSTM, self).__init__()

        # load the configuration file
        config = load_config()
        model_config = config['model']
        data_config = config['data']

        # define the model configuration (+ num_keys)
        self.input_size = input_size if input_size is not None else model_config['input_size']
        self.hidden_size = hidden_size if hidden_size is not None else model_config['hidden_size']
        self.num_layers = num_layers if num_layers is not None else model_config['num_layers']
        self.bias = bias if bias is not None else model_config['bias']
        self.batch_first = batch_first if batch_first is not None else model_config['batch_first']
        self.dropout = dropout if dropout is not None else model_config['dropout']
        self.bidirectional = bidirectional if bidirectional is not None else model_config['bidirectional']
        self.proj_size = proj_size if proj_size is not None else model_config['proj_size']
        self.num_keys = num_keys if num_keys is not None else data_config['num_keys']

        # instantiate the LSTM model
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.dropout,
            self.bidirectional,
            self.proj_size
        )

        # fully-connected layer
        self.fc = nn.Linear(self.hidden_size, self.num_keys)

        # softmax to rescale an n-dimensional input Tensor
        # ensuring values within [0,1] and sum to 1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Method to perform the forward pass through the LSTM.
        :param self: LSTM model
        :param x: [batch_size, seq_len, input_size]
        :return:
        """
        # calculate the LSTM output
        lstm_out, _ = self.lstm(x)

        # pass through the fully connected layer to get logits
        logits = self.fc(lstm_out[:, -1, :])

        # apply softmax to get probabilities
        probabilities = self.softmax(logits)

        return probabilities
