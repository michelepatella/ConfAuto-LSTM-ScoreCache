import logging
import torch
import numpy as np
from utils.AccessLogsDataset import AccessLogsDataset
from utils.LSTM import LSTM
from utils.config_utils import _get_config_value
from utils.dataset_utils import _get_dataset_path_type, _create_data_loader
from utils.training_utils import _build_optimizer
from sklearn.utils.class_weight import compute_class_weight


def _training_testing_setup(
        model_params,
        learning_rate,
        targets
):
    """
    Method to set up the training and testing processes.
    :param model_params: The model parameters.
    :param learning_rate: The learning rate.
    :param targets: The targets.
    :return: The device to use, the loss function, the model and the optimizer.
    """
    # initial message
    logging.info("🔄 Training/Testing setup started...")

    # debugging
    logging.debug(f"⚙️ Model params: {model_params}.")
    logging.debug(f"⚙️ Learning rate: {learning_rate}.")
    logging.debug(f"⚙️ Targets: {targets}.")

    try:
        # define the device to use
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")

        # get the class weights
        class_weights = _calculate_class_weights(targets)

        # debugging
        logging.debug(f"⚙️ Class weights: {class_weights}.")

        # define the loss function
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights).
            float().to(device)
        )

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


def _loader_setup(loader_type, shuffle):
    """
    Method to prepare the data loader for the training and testing.
    :param loader_type: The loader type ("training" or "testing").
    :param shuffle: Whether to shuffle the data.
    :return: The created data loader and the corresponding dataset.
    """
    # get the dataset type
    dataset_path, _ = _get_dataset_path_type()

    # debugging
    logging.debug(f"⚙️ Loader type: {loader_type}.")
    logging.debug(f"⚙️ Shuffle: {shuffle}.")

    # get the dataset
    dataset = AccessLogsDataset(
        _get_config_value(dataset_path),
        loader_type
    )

    # create the data loader starting from the dataset
    loader = _create_data_loader(
        dataset,
        _get_config_value(f"{loader_type}.batch_size"),
        shuffle
    )

    return dataset, loader


def _extract_targets_from_loader(data_loader):
    """
    Method to extract the targets from the data loader.
    :param data_loader: The data loader from which to extract the targets.
    :return: All the extracted targets.
    """
    # initial message
    logging.info("🔄 Target extraction from loader started...")

    try:
        all_targets = []
        # extract targets from data loader
        for _, targets in data_loader:
            all_targets.append(targets - 1)

    except Exception as e:
        raise Exception(f"❌ Error while extracting targets from loader: {e}")

    # debugging
    logging.debug(f"⚙️ Target extracted: {all_targets}.")

    # show a successful message
    logging.info("🟢 Target extracted from loader.")

    return torch.cat(all_targets)


def _calculate_class_weights(targets):
    """
    Method to calculate the class weights.
    :param targets: The targets for which to calculate the class weights.
    :return: The class weights calculated.
    """
    # initial message
    logging.info("🔄 Class weights calculation started...")

    try:
        # get the tot. no. of classes
        num_classes = _get_config_value("data.num_keys")

        # debugging
        logging.debug(f"⚙️ Number of classes: {num_classes}.")

        # be sure targets is a numpy array and shift them
        targets = targets.cpu().numpy() if (
            isinstance(targets, torch.Tensor)) \
            else targets

        # get the classes appearing in target list
        present_classes = np.unique(targets)

        # debugging
        logging.debug(f"⚙️ Present classes: {present_classes}.")

        # compute the class weights
        computed_weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=targets
        )

        # initialize weights to 1.0
        class_weights = np.ones(num_classes, dtype=np.float32)

        # update weights for appearing classes
        for cls, weight in zip(present_classes, computed_weights):
            class_weights[cls] = weight

    except Exception as e:
        raise Exception(f"❌ Error while calculating the class weights: {e}")

    # show a successful message
    logging.info("🟢 Class weights calculated.")

    return class_weights