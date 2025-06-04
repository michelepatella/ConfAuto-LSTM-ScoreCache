from utils.data.dataloader.dataloader_utils import extract_targets_from_dataloader
from utils.logs.log_utils import info
from utils.model.setup.model_setup import model_setup
from utils.model.setup.model_loader import load_model


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