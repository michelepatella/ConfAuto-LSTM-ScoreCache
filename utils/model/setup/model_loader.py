import torch
from utils.logs.log_utils import info, debug


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