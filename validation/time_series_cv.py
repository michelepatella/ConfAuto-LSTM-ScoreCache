import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from model.lstm_model import LSTM
from utils.config_loader import load_config
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
    # load training and validation configurations
    config = load_config()
    training_config = config["training"]
    validation_config = config["validation"]

    # extract labels from the dataset
    labels = np.array([label for _, label in dataset])

    # setup for the TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=validation_config["num_folds"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in tscv.split(np.zeros(len(labels))):

        # define training and validation sets
        training_dataset = Subset(dataset, train_idx)
        validation_dataset = Subset(dataset, val_idx)

        # define train and validation loader
        training_loader = DataLoader(
            training_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=training_config["batch_size"]
        )

        # define the custom model
        model = LSTM(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # optimize to accelerate the learning process
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # train the model
        model.train()

        for x, y in training_loader:

            optimizer.zero_grad()

            # unpack
            x_keys, x_timestamps, x_features = x
            x_keys = x_keys.to(device)
            x_timestamps = x_timestamps.to(device)
            x_features = x_features.to(device)
            y = y.to(device)

            # calculate the outputs
            outputs = model(x_features, x_timestamps, x_keys)

            # calculate the loss
            loss = criterion(outputs, y)

            # perform the backward pass
            loss.backward()

            # optimize
            optimizer.step()

        # evaluate the model
        val_loss = _evaluate_model(model, validation_loader, criterion, device)
        fold_losses.append(val_loss)

    return fold_losses