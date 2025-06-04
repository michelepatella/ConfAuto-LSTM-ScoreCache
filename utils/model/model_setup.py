import torch
from utils.data.dataloader.dataloader_utils import extract_targets_from_dataloader
from utils.logs.log_utils import info, debug
from utils.model.LSTM import LSTM
from utils.model.model_utils import calculate_class_weights, build_optimizer


def load_model(
        model,
        device,
        config_settings
):
    """
    Method to load a model.
    :param model: The initialization of the model.
    :param device: The device to use.
    :param config_settings: The configuration settings.
    :return: The model loaded.
    """
    # initial message
    info("🔄 Model loading started...")

    # debugging
    debug(f"⚙️ Path to load the model: {config_settings.model_save_path}.")

    try:
        # load the model
        model.load_state_dict(torch.load(
            config_settings.model_save_path,
            map_location=device
        ))
    except (
            FileNotFoundError,
            PermissionError,
            AttributeError,
            ValueError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while loading the model: {e}.")

    # show a successful message
    info("🟢 Model loaded.")

    return model


def trained_model_setup(
        loader,
        config_settings
):
    """
    Method to set up the trained model.
    :param loader: The loader to be used.
    :param config_settings: The configuration settings.
    :return: The device to use, the loss function, and the model.
    """
    # initial message
    info("🔄 Trained model setup started...")

    # setup for the model
    device, criterion, model, _ = (
        model_setup(
            config_settings.model_params,
            config_settings.learning_rate,
            extract_targets_from_dataloader(loader),
            config_settings
        )
    )

    # load the trained model
    model = load_model(
        model,
        device,
        config_settings
    )

    # show a successful message
    info("🟢 Trained model setup completed.")

    return (
        device,
        criterion,
        model
    )


def model_setup(
        model_params,
        learning_rate,
        targets,
        config_settings
):
    """
    Method to set up the training and testing processes.
    :param model_params: The model parameters.
    :param learning_rate: The learning rate.
    :param targets: The targets.
    :param config_settings: The configuration settings.
    :return: The device to use, the loss function, the model and the optimizer.
    """
    # initial message
    info("🔄 Model setup started...")

    # debugging
    debug(f"⚙️ Model params: {model_params}.")
    debug(f"⚙️ Learning rate: {learning_rate}.")
    debug(f"⚙️ Targets: {targets}.")

    try:
        # define the device to use
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        # get the class weights
        class_weights = calculate_class_weights(
            targets,
            config_settings
        )

        # debugging
        debug(f"⚙️ Class weights: {class_weights}.")

        # define the loss function
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights).
            float().to(device)
        )

        # define the LSTM model
        model = (
            LSTM(
                model_params,
                config_settings
            ).to(device)
        )

        # define the optimizer
        optimizer = build_optimizer(
            model,
            learning_rate,
            config_settings
        )
    except (
            TypeError,
            ValueError,
            KeyError
    ) as e:
        raise RuntimeError(f"❌ Error while setting up the model: {e}.")

    # show a successful message
    info("🟢 Model setup completed.")

    return (
        device,
        criterion,
        model,
        optimizer
    )