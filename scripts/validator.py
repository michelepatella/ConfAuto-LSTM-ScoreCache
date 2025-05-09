import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from model.lstm_model import LSTM
from utils.access_log_dataset import AccessLogsDataset
from utils.config_loader import load_config

def evaluate_model(model, loader, criterion, device):
    """
    Evaluate a model on a validation dataset.
    :param model: the model to evaluate
    :param loader: the validation loader
    :param criterion: the loss criterion
    :param device: device to use
    :return: the average loss
    """
    # evaluate the model
    model.eval()

    # initialize the total loss
    total_loss = 0.0

    # calculate the average loss
    with torch.no_grad():
        for x, y in loader:

            x, y = x.to(device), y.to(device)

            # calculate the outputs
            outputs = model(x)

            # calculate the loss and update the total one
            loss = criterion(outputs, y)
            total_loss += loss.item()

    # return the average loss
    return total_loss / len(loader)

def stratified_cv(
        labels,
        dataset,
        embedding_dim,
        hidden_size,
        num_layers,
        dropout,
        lr,
        criterion,
        fold_losses
):
    # load training configuration
    config = load_config()
    training_config = config['training']

    # setup for the 10-fold CV
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):

        # define training and validation sets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=training_config['batch_size'], shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'])

        # define the custom model
        model = LSTM(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # optimize to accelerate the learning process
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # train the model
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # calculate the outputs
            outputs = model(x)

            # calculate the loss
            loss = criterion(outputs, y)

            # perform the backward pass
            loss.backward()

            # optimize
            optimizer.step()

        # evaluate the model
        val_loss = evaluate_model(model, val_loader, criterion, device)
        fold_losses.append(val_loss)

    return fold_losses

def parameter_tuning():
    """
    Method to perform parameter tuning.
    :return:
    """
    # load data configuration
    config = load_config()
    data_config = config['data']

    # load the dataset and extract the labels
    dataset = AccessLogsDataset(data_config['static_dataset_path'])
    labels = np.array([label for _, label in dataset])

    # define the loss function
    criterion = nn.CrossEntropyLoss()

    best_params = {}
    best_avg_loss = float("inf")

    # grid search to find best params
    for embedding_dim in [32, 64]:
        for hidden_size in [64, 128]:
            for num_layers in [1, 3]:
                for dropout in [0.0, 0.5]:
                    for lr in [0.001, 0.0005]:
                        fold_losses = []

                        # loop over all the folders of the CV
                        fold_losses = stratified_cv(
                            labels,
                            dataset,
                            embedding_dim,
                            hidden_size,
                            num_layers,
                            dropout,
                            lr,
                            criterion,
                            fold_losses
                        )

                        # calculate and show the average loss
                        avg_loss = np.mean(fold_losses)
                        print(f"[CV] Params: emb={embedding_dim}, hid={hidden_size}, "
                              f"layers={num_layers}, drop={dropout}, lr={lr} -> avg_loss={avg_loss}")

                        # if the average loss is less than the best one, update it
                        if avg_loss < best_avg_loss:

                            # update the best loss
                            best_avg_loss = avg_loss

                            # save best params
                            best_params = {
                                "embedding_dim": embedding_dim,
                                "hidden_size": hidden_size,
                                "num_layers": num_layers,
                                "dropout": dropout,
                                "learning_rate": lr
                            }

    # print the best params
    print("Best params from CV:", best_params)

    # update the best params
    config['model'].update({
        "embedding_dim": best_params["embedding_dim"],
        "hidden_size": best_params["hidden_size"],
        "num_layers": best_params["num_layers"],
        "dropout": best_params["dropout"]
    })
    config['training']['learning_rate'] = best_params["learning_rate"]

    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    print("Updated config saved to 'config.yaml'")