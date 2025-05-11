import logging
import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.model_selection import TimeSeriesSplit
from model.LSTM import LSTM
from utils.config_utils import load_config
from utils.data_loader_utils import _create_data_loader
from utils.evaluation_utils import _evaluate_model
from utils.training_utils import _train_one_epoch


def _time_series_cv(
        dataset,
        hidden_size,
        num_layers,
        dropout,
        learning_rate,
        criterion,
        fold_losses
):
    """
    Method to perform time series cross-validation.
    :param dataset: The dataset to use.
    :param hidden_size: The hidden dimension.
    :param num_layers: The number of layers.
    :param dropout: The dropout rate.
    :param learning_rate: The learning rate.
    :param criterion: The loss function.
    :param fold_losses: The fold_losses.
    :return: The updated fold losses as output.
    """
    # load config file
    config = load_config()
    training_config = config["training"]
    validation_config = config["validation"]

    # get the no. of samples in the dataset
    n_samples = len(dataset)

    # try setup for the TimeSeriesSplit
    try:
        # setup
        tscv = TimeSeriesSplit(n_splits=validation_config["num_folds"])
    except Exception as e:
        raise Exception(f"An unexpected error while instantiating Time Series Split: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in tscv.split(np.arange(n_samples)):

        # define training and validation sets
        training_dataset = Subset(dataset, train_idx)
        validation_dataset = Subset(dataset, val_idx)

        # create training and validation loaders
        training_loader = _create_data_loader(
            training_dataset,
            training_config["batch_size"]
        )
        validation_loader = _create_data_loader(
            validation_dataset,
            training_config["batch_size"]
        )

        # try to define the LSTM model
        try:
            # define the LSTM model
            model = LSTM(
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        except Exception as e:
            raise Exception(f"An unexpected error while loading model: {e}")

        # optimize to accelerate the learning process
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # train the model (only once)
        _train_one_epoch(model, training_loader, optimizer, criterion, device)

        # evaluate the model
        val_loss = _evaluate_model(model, validation_loader, criterion, device)

        # check the val_loss and update fold losses
        if val_loss is not None:
            fold_losses.append(val_loss)
        else:
            logging.warning("Validation returned None. Skipping this fold.")

    return fold_losses