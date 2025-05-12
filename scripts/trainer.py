import torch
import torch.nn as nn
import logging
from model.LSTM import LSTM
from utils.AccessLogsDataset import AccessLogsDataset
from utils.config_utils import load_config, get_config_value
from utils.dataset_utils import _create_data_loader
from utils.training_utils import _train_one_epoch


def train_model():
    """
    Method to train the LSTM model.
    :return:
    """
    # load config file
    config = load_config()

    # load the dataset
    dataset = AccessLogsDataset(
        get_config_value(config, "data.static_dataset_path"),
    "training"
    )

    # create the training loader
    training_loader = _create_data_loader(
        dataset,
        get_config_value(config, "training.batch_size")
    )

    # select the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the LSTM model
    model = LSTM().to(device)

    # definition of the loss function
    criterion = nn.CrossEntropyLoss()

    try:
        # define optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=get_config_value(config, "training.learning_rate")
        )
    except Exception as e:
        raise Exception("Error while defining the optimizer.")

    # train the model
    for epoch in range(get_config_value(config, "training.epochs")):

        logging.info(f"Epoch {epoch + 1}/{get_config_value(
            config, 
            'training.epochs'
        )}")

        # train the model
        _train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device
        )

    # save the trained model
    torch.save(
        model.state_dict(),
        get_config_value(config, "model.model_save_path")
    )