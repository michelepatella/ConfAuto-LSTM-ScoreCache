import torch
from utils.logs.log_utils import info, debug


def save_model(
        model,
        config_settings
):
    """
    Method to save a model.
    :param model: The model to be saved.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Model saving started...")

    try:
        # debugging
        debug(f"⚙️ Path to save the model: {config_settings.model_save_path}.")

        # save the model
        torch.save(
            model.state_dict(),
            config_settings.model_save_path
        )
    except (
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            FileNotFoundError,
            PermissionError
    ) as e:
        raise RuntimeError(f"❌ Error while saving the model: {e}.")

    # show a successful message
    info(f"🟢 Model save to '{config_settings.model_save_path}'.")