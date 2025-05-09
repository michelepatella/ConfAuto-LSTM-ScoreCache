import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from model.lstm_model import LSTM
from utils.access_log_dataset import AccessLogsDataset
from utils.config_loader import load_config

def evaluate_model(model, loader, criterion, device):
    """
    Evaluate a model on a validation dataset.
    :param model: The model to evaluate.
    :param loader: The validation loader.
    :param criterion: The loss function.
    :param device: Device to use.
    :return: The average loss.
    """
    # evaluate the model
    model.eval()

    # initialize the total loss
    total_loss = 0.0

    # calculate the average loss
    with torch.no_grad():
        for x, y in loader:

            # unpack
            x_keys, x_timestamps = x
            x_keys, x_timestamps, y = x_keys.to(device), x_timestamps.to(device), y.to(device)

            # calculate the outputs
            outputs = model(x_keys, x_timestamps)

            # calculate the loss and update the total one
            loss = criterion(outputs, y)
            total_loss += loss.item()

    # return the average loss
    return total_loss / len(loader)

def stratified_cv(
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

    # setup for the 10-folds CV
    skf = StratifiedKFold(
        n_splits=validation_config["num_folds"],
        shuffle=True,
        random_state=validation_config["random_state"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):

        # define training and validation sets
        training_dataset = Subset(dataset, train_idx)
        validation_dataset = Subset(dataset, val_idx)

        # define train and validation loader
        training_loader = DataLoader(
            training_dataset,
            batch_size=training_config["batch_size"],
            shuffle=True
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
            x_keys, x_timestamps = x
            x_keys, x_timestamps, y = x_keys.to(device), x_timestamps.to(device), y.to(device)

            # calculate the outputs
            outputs = model(x_keys, x_timestamps)

            # calculate the loss
            loss = criterion(outputs, y)

            # perform the backward pass
            loss.backward()

            # optimize
            optimizer.step()

        # evaluate the model
        val_loss = evaluate_model(model, validation_loader, criterion, device)
        fold_losses.append(val_loss)

    return fold_losses

def set_best_params(best_params):
    """
    Method to save the best parameters into a configuration file.
    :param best_params: Best parameters found.
    :return:
    """
    # load configuration
    config = load_config()

    # print the best params found
    print("Best params from Stratified 10-folds CV:", best_params)

    # update the best params
    config["model"].update({
        "embedding_dim": best_params["embedding_dim"],
        "hidden_size": best_params["hidden_size"],
        "num_layers": best_params["num_layers"],
        "dropout": best_params["dropout"]
    })
    config["training"].update({
        "learning_rate": best_params["learning_rate"],
    })

def check_and_update_best_params(fold_losses, best_avg_loss, curr_params, best_params):
    """
    Method to calculate the average loss and update the best parameters
    in case the average loss is less than the current best loss.
    :param fold_losses: The loss of the current fold iteration.
    :param best_avg_loss: The current best average loss.
    :param curr_params: The current parameters (used in the current fold iteration).
    :param best_params: The current best parameters.
    :return: The best average loss and the best parameters as output.
    """
    # calculate the average loss
    avg_loss = np.mean(fold_losses)

    # if the average loss is less than the best one,
    # update it and the best params
    if avg_loss < best_avg_loss:
        # update the best loss
        best_avg_loss = avg_loss

        # update the best params
        best_params = {
            "embedding_dim": curr_params["embedding_dim"],
            "hidden_size": curr_params["hidden_size"],
            "num_layers": curr_params["num_layers"],
            "dropout": curr_params["dropout"],
            "learning_rate": curr_params["learning_rate"]
        }

    return best_avg_loss, best_params

def grid_search(dataset, criterion):
    """
    Method to perform grid search to find the best parameters.
    :param dataset: The dataset on which to work on.
    :param criterion: The loss function.
    :return: The best parameters as output.
    """
    # load validation configuration
    config = load_config()
    validation_config = config["validation"]

    # initialize the best parameters and average loss
    best_params = {}
    best_avg_loss = float("inf")

    # define the parameters combination
    param_combinations = [
        (embedding_dim, hidden_size, num_layers, dropout, learning_rate)
        for embedding_dim in validation_config["embedding_dim_range"]
        for hidden_size in validation_config["hidden_size_range"]
        for num_layers in validation_config["num_layers_range"]
        for dropout in validation_config["dropout_range"]
        for learning_rate in validation_config["learning_rate_range"]
    ]

    # grid search
    with tqdm(total=len(param_combinations), desc="Grid Search Progress") as pbar:
        for embedding_dim, hidden_size, num_layers, dropout, learning_rate in param_combinations:

            fold_losses = []

            # perform stratified 10-folds CV
            fold_losses = stratified_cv(
                dataset,
                embedding_dim,
                hidden_size,
                num_layers,
                dropout,
                learning_rate,
                criterion,
                fold_losses
            )

            # group current parameters together
            curr_params = {
                "embedding_dim": embedding_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": learning_rate
            }

            # check the loss and eventually update the best parameters
            best_avg_loss, best_params = check_and_update_best_params(
                fold_losses,
                best_avg_loss,
                curr_params,
                best_params
            )

            # update the progress bar
            pbar.update(1)
    print("Best avg loss:", best_avg_loss)
    return best_params

def parameter_tuning():
    """
    Method to orchestrate the parameter tuning of the model.
    :return:
    """
    # load data configuration
    config = load_config()
    data_config = config["data"]

    # load the dataset
    dataset = AccessLogsDataset(data_config["static_dataset_path"], "validation")

    # define the loss function
    criterion = nn.CrossEntropyLoss()

    # grid search for best parameters
    best_params = grid_search(dataset, criterion)

    # set the best parameters
    set_best_params(best_params)

parameter_tuning()