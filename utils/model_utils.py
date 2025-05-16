import numpy as np
import torch
from sklearn.utils import compute_class_weight
from utils.LSTM import LSTM
from utils.log_utils import _info, _debug
from utils.config_utils import _get_config_value
from utils.training_utils import _build_optimizer


def _save_model(model):
    """
    Method to save a model.
    :param model: The model to be saved.
    :return:
    """
    # initial message
    _info("🔄 Model saving started...")

    try:
        # get the model path
        model_path = _get_config_value("model.model_save_path")

        # debugging
        _debug(f"⚙️ Path to save the model: {model_path}.")

        # save the model
        torch.save(
            model.state_dict(),
            model_path
        )
    except Exception as e:
        raise Exception(f"❌ Error while saving the model: {e}")

    # show a successful message
    _info(f"🟢 Model save to '{model_path}'.")


def _load_model(model, model_path, device):
    """
    Method to load a model.
    :param model: The initialization of the model.
    :param model_path: The path of the model.
    :param device: The device to use.
    :return: The model loaded.
    """
    # initial message
    _info("🔄 Model loading started...")

    # debugging
    _debug(f"⚙️ Path to load the model: {model_path}.")

    try:
        # load the model
        model.load_state_dict(torch.load(
            model_path,
            map_location=device
        ))
    except Exception as e:
        raise Exception(f"❌ Error while loading the model: {e}")

    # show a successful message
    _info("🟢 Model loaded.")

    return model


def _model_setup(
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
    _info("🔄 Training/Testing setup started...")

    # debugging
    _debug(f"⚙️ Model params: {model_params}.")
    _debug(f"⚙️ Learning rate: {learning_rate}.")
    _debug(f"⚙️ Targets: {targets}.")

    try:
        # define the device to use
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")

        # get the class weights
        class_weights = _calculate_class_weights(targets)

        # debugging
        _debug(f"⚙️ Class weights: {class_weights}.")

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
    _info("🟢 Training/Testing setup completed.")

    return device, criterion, model, optimizer


def _calculate_class_weights(targets):
    """
    Method to calculate the class weights.
    :param targets: The targets for which to calculate the class weights.
    :return: The class weights calculated.
    """
    # initial message
    _info("🔄 Class weights calculation started...")

    try:
        # get the tot. no. of classes
        num_classes = _get_config_value("data.num_keys")

        # debugging
        _debug(f"⚙️ Number of classes: {num_classes}.")

        # be sure targets is a numpy array and shift them
        targets = targets.cpu().numpy() if (
            isinstance(targets, torch.Tensor)) \
            else targets

        # get the classes appearing in target list
        present_classes = np.unique(targets)

        # debugging
        _debug(f"⚙️ Present classes: {present_classes}.")

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
    _info("🟢 Class weights calculated.")

    return class_weights
