import logging
import torch
from utils.config_utils import _get_config_value


def _save_model(model):
    """
    Method to save a model.
    :param model: The model to be saved.
    :return:
    """
    # initial message
    logging.info("🔄 Model saving started...")

    try:
        # get the model path
        model_path = _get_config_value("model.model_save_path")

        # save the model
        torch.save(
            model.state_dict(),
            model_path
        )
    except Exception as e:
        raise Exception(f"❌ Error while saving the model: {e}")

    # show a successful message
    logging.info(f"🟢 Model save to '{model_path}'.")


def _load_model(model, model_path, device):
    """
    Method to load a model.
    :param model: The initialization of the model.
    :param model_path: The path of the model.
    :param device: The device to use.
    :return: The model loaded.
    """
    # initial message
    logging.info("🔄 Model loading started...")

    try:
        # load the model
        model.load_state_dict(torch.load(
            model_path,
            map_location=device
        ))
    except Exception as e:
        raise Exception(f"❌ Error while loading the model: {e}")

    # show a successful message
    logging.info("🟢 Model loaded.")

    return model
