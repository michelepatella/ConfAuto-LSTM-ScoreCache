import logging
import torch
from utils.AccessLogsDataset import AccessLogsDataset
from utils.LSTM import LSTM
from utils.config_utils import _get_config_value
from utils.dataset_utils import _get_dataset_path_type, _create_data_loader
from utils.training_utils import _build_optimizer


def _training_testing_setup(model_params, learning_rate):
    """
    Method to set up the training and testing processes.
    :param model_params: The model parameters.
    :param learning_rate: The learning rate.
    :return: The device to use, the loss function, the model and the optimizer.
    """
    # initial message
    logging.info("🔄 Training/Testing setup started...")

    try:
        # define the device to use
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")

        # define the loss function
        criterion = torch.nn.CrossEntropyLoss()

        # define the LSTM model
        model = LSTM(model_params).to(device)

        # define the optimizer
        optimizer = _build_optimizer(
            model,
            learning_rate
        )
    except Exception as e:
        raise Exception(f"❌ Error while setting up the training/testing process: {e}")

    # show a successful message
    logging.info("🟢 Training/Testing setup completed.")

    return device, criterion, model, optimizer


def _loader_setup(loader_type):
    """
    Method to prepare the data loader for the training and testing.
    :param loader_type: The loader type ("training" or "testing").
    :return: The created data loader.
    """
    # get the dataset type
    dataset_path, _ = _get_dataset_path_type()

    # get the dataset
    dataset = AccessLogsDataset(
        _get_config_value(dataset_path),
        loader_type
    )

    # create the data loader starting from the dataset
    loader = _create_data_loader(
        dataset,
        _get_config_value(f"{loader_type}.batch_size")
    )

    return loader