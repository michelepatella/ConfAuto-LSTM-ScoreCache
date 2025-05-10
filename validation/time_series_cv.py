import logging
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from model.lstm_model import LSTM
from utils.config_loader import load_config
from validation.batch_processor import _process_batch
from validation.model_evaluator import _evaluate_model


def _time_series_cv(
        dataset,
        embedding_dim,
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
    :param embedding_dim: The embedding dimension.
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

    # load configs
    if (config is not None and ("training" in config)
            and ("validation" in config)):
        training_config = config["training"]
        validation_config = config["validation"]
    else:
        return fold_losses

    # try to extract labels
    try:
        # extract labels
        labels = np.array([label for _, label in dataset])
    except Exception as e:
        logging.error(f"An unexpected error while extracting labels from dataset: {e}")
        return fold_losses

    # try setup for the TimeSeriesSplit
    try:
        # setup
        tscv = TimeSeriesSplit(n_splits=validation_config["num_folds"])
    except KeyError as e:
        logging.error(f"Missing key in validation config: {e}")
        return fold_losses
    except Exception as e:
        logging.error(f"An unexpected error while instantiating Time Series Split: {e}")
        return fold_losses

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in tscv.split(np.zeros(len(labels))):

        # define training and validation sets
        training_dataset = Subset(dataset, train_idx)
        validation_dataset = Subset(dataset, val_idx)

        # try to define training loader
        try:
            # define training loader
            training_loader = DataLoader(
                training_dataset,
                batch_size=training_config["batch_size"],
                shuffle=False
            )
        except Exception as e:
            logging.error(f"An unexpected error while loading training data: {e}")
            return fold_losses

        # try to define validation loader
        try:
            # define training loader
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=training_config["batch_size"]
            )
        except Exception as e:
            logging.error(f"An unexpected error while loading validation data: {e}")
            return fold_losses

        # try to define the LSTM model
        try:
            # define the LSTM model
            model = LSTM(
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        except Exception as e:
            logging.error(f"An unexpected error while loading model: {e}")
            return fold_losses

        # optimize to accelerate the learning process
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # train the model
        model.train()

        for x, y in training_loader:

            optimizer.zero_grad()

            # calculate loss by processing the batch
            loss = _process_batch((x, y), model, criterion, device)

            # check loss
            if loss is None:
                return fold_losses

            # try to perform backward pass
            try:
                # backward pass with optimization
                loss.backward()
                optimizer.step()
            except Exception as e:
                logging.error(f"An unexpected error during backpropagation: {e}")
                return fold_losses

        # evaluate the model
        val_loss = _evaluate_model(model, validation_loader, criterion, device)

        # check the val_loss and update fold losses
        if val_loss is not None:
            fold_losses.append(val_loss)
        else:
            logging.warning("Validation returned None. Skipping this fold.")

    return fold_losses