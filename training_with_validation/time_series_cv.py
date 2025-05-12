import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.model_selection import TimeSeriesSplit
from model.LSTM import LSTM
from utils.config_utils import load_config, get_config_value
from utils.dataset_utils import _create_data_loader
from utils.evaluation_utils import _evaluate_model
from utils.training_utils import _train_one_epoch


def _time_series_cv(
        training_set,
        hidden_size,
        num_layers,
        dropout,
        learning_rate,
        fold_losses
):
    """
    Method to perform time series cross-validation.
    :param training_set: The training set.
    :param hidden_size: The hidden dimension.
    :param num_layers: The number of layers.
    :param dropout: The dropout rate.
    :param learning_rate: The learning rate.
    :param fold_losses: The fold_losses.
    :return: The updated fold losses.
    """
    # load config file
    config = load_config()

    # get the no. of samples in the dataset
    n_samples = len(training_set)

    # try setup for the TimeSeriesSplit
    try:
        # setup
        tscv = TimeSeriesSplit(n_splits=get_config_value(
            config,
            "validation.num_folds"
        ))
    except Exception as e:
        raise Exception(f"Error while instantiating Time Series Split: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in tscv.split(np.arange(n_samples)):

        try:
            # define training and validation sets
            training_dataset = Subset(training_set, train_idx)
            validation_dataset = Subset(training_set, val_idx)
        except Exception as e:
            raise Exception(f"Error while defining training and validation sets: {e}")

        # create training and validation loaders
        training_loader = _create_data_loader(
            training_dataset,
            get_config_value(config, "training.batch_size")
        )
        validation_loader = _create_data_loader(
            validation_dataset,
            get_config_value(config, "training.batch_size")
        )

        # define the LSTM model
        model = LSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        try:
            # optimize
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )
        except Exception as e:
            raise Exception(f"Error while instantiating optimizer: {e}")

        # train the model (only once)
        _train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device
        )

        # evaluate the model
        val_loss = _evaluate_model(
            model,
            validation_loader,
            criterion,
            device
        )

        # check the val_loss and update fold losses
        if val_loss is not None:
            fold_losses.append(val_loss)
        else:
            raise Exception("Validation loss returned None.")

    return fold_losses