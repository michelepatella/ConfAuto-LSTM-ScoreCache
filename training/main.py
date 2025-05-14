import torch
import torch.nn as nn
import logging
from model.LSTM import LSTM
from utils.AccessLogsDataset import AccessLogsDataset
from utils.config_utils import _get_config_value
from utils.dataset_utils import _create_data_loader, _get_dataset_path_type
from utils.training_utils import _train_one_epoch, _build_optimizer


def training():
    """
    Method to train the LSTM model.
    :return:
    """
    # get the dataset path
    dataset_path, _ = _get_dataset_path_type()

    # load the dataset
    dataset = AccessLogsDataset(
        _get_config_value(dataset_path),
    "training"
    )

    # create the training loader
    training_loader = _create_data_loader(
        dataset,
        _get_config_value("training.batch_size")
    )

    # select the device to use
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    # define the LSTM model
    model = LSTM(_get_config_value("model.params")).to(device)

    # definition of the loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = _build_optimizer(
        model,
        _get_config_value("training.learning_rate")
    )

    # train the model
    for epoch in range(_get_config_value("training.epochs")):

        logging.info(f"Epoch {epoch + 1}/{_get_config_value('training.epochs')}")

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
        _get_config_value("model.model_save_path")
    )