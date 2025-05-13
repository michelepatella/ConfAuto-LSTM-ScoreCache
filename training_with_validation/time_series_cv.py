import numpy as np
import logging
import torch
from torch.utils.data import Subset
from sklearn.model_selection import TimeSeriesSplit
from model.LSTM import LSTM
from utils.config_utils import _get_config_value
from utils.dataset_utils import _create_data_loader
from utils.evaluation_utils import _evaluate_model
from utils.training_utils import _train_one_epoch


def _time_series_cv(
        training_set,
        hidden_size,
        num_layers,
        dropout,
        learning_rate
):
    """
    Method to perform Time Series cross-validation.
    :param training_set: The training set.
    :param hidden_size: The hidden dimension.
    :param num_layers: The number of layers.
    :param dropout: The dropout rate.
    :param learning_rate: The learning rate.
    :return: The final average loss.
    """
    # initial message
    logging.info("🔄 Time Series started...")

    # get the no. of samples in the dataset
    n_samples = len(training_set)

    try:
        # setup for Time Series Split
        tscv = TimeSeriesSplit(n_splits=_get_config_value(
            "validation.num_folds"
        ))
    except Exception as e:
        raise Exception(f"❌ Error while instantiating Time Series Split: {e}")

    # define the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    fold_losses = []
    # iterate over the training set
    for train_idx, val_idx in tscv.split(np.arange(n_samples)):

        try:
            # define training and validation sets
            training_dataset = Subset(training_set, train_idx)
            validation_dataset = Subset(training_set, val_idx)
        except Exception as e:
            raise Exception(f"❌ Error while defining training and validation sets: {e}")

        # create training and validation loaders
        training_loader = _create_data_loader(
            training_dataset,
            _get_config_value("training.batch_size")
        )
        validation_loader = _create_data_loader(
            validation_dataset,
            _get_config_value("training.batch_size")
        )

        # define the LSTM model
        model = LSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        try:
            # define the optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )
        except Exception as e:
            raise Exception(f"❌ Error while instantiating optimizer: {e}")

        # train the model
        for epoch in range(_get_config_value("validation.epochs")):
            _train_one_epoch(
                model,
                training_loader,
                optimizer,
                criterion,
                device
            )

        # evaluate the model (by only average loss, no metrics)
        avg_loss, _ = _evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
            compute_metrics=True
        )
        fold_losses.append(avg_loss)

    logging.info("🟢 Time Series CV completed.")

    # calculate the average of the loss
    final_avg_loss = np.mean(fold_losses)

    return final_avg_loss