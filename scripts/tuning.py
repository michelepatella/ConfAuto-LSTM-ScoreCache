from model.lstm_model import LSTM
from utils.access_log_dataset import AccessLogsDataset
from utils.config_loader import load_config
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def evaluate_model(model, dataloader, criterion, device):
    """
    Method to evaluate the average loss across the dataset.
    :param model: The LSTM model
    :param dataloader: DataLoader object
    :param criterion: Loss function
    :param device: Device used for the model
    :return: The average loss across the dataset
    """
    # evaluate the model
    model.eval()

    # initialize the total loss
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # calculate the outputs
            outputs = model(x)

            # calculate the loss and update it
            loss = criterion(outputs, y)
            total_loss += loss.item()

    # return the average loss across the dataset
    return total_loss / len(dataloader)

def parameter_tuning():
    """
    Method to tune the parameters of the LSTM model.
    :return:
    """
    # load data and training configuration
    config = load_config()
    data_config = config['data']
    training_config = config['training']

    # load the validation dataset
    dataset = AccessLogsDataset(data_config['static_val_path'])
    dataloader = DataLoader(dataset, batch_size=training_config['batch_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the loss function
    criterion = nn.CrossEntropyLoss()

    best_params = {}
    best_loss = float("inf")

    # grid search
    for embedding_dim in [32, 64]:
        for hidden_size in [64, 128]:
            for num_layers in [1, 3]:
                for dropout in [0.0, 0.5]:
                    for lr in [0.001, 0.0005]:

                        # instantiate the LSTM model
                        model = LSTM(
                            embedding_dim=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout
                        ).to(device)

                        # to accelerate the learning process
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                        # train the model via a single epoch
                        model.train()
                        for x, y in dataloader:
                            x, y = x.to(device), y.to(device)
                            optimizer.zero_grad()

                            # get the outputs of the model
                            outputs = model(x)

                            # calculate the loss
                            loss = criterion(outputs, y)

                            # perform backward pass
                            loss.backward()

                            # optimize
                            optimizer.step()

                        # evaluate the model
                        val_loss = evaluate_model(model, dataloader, criterion, device)
                        print(f"Params: emb={embedding_dim}, hid={hidden_size}, layers={num_layers}, drop={dropout}, lr={lr} -> loss={val_loss}")

                        # check whether is the best loss
                        if val_loss < best_loss:
                            # update the best loss
                            best_loss = val_loss

                            # store the best hyperparameters
                            best_params = {
                                "embedding_dim": embedding_dim,
                                "hidden_size": hidden_size,
                                "num_layers": num_layers,
                                "dropout": dropout,
                                "learning_rate": lr
                            }

    # show best hyperparameters
    print("Best params:", best_params)

    # save the best hyperparameters in the config file
    config = load_config()
    config['model'].update({
        "embedding_dim": best_params["embedding_dim"],
        "hidden_size": best_params["hidden_size"],
        "num_layers": best_params["num_layers"],
        "dropout": best_params["dropout"]
    })
    config['training']['learning_rate'] = best_params["learning_rate"]

parameter_tuning()