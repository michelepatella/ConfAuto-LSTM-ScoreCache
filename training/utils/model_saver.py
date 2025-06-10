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
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"FileNotFoundError: {e}.")
    except PermissionError as e:
        raise PermissionError(f"PermissionError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info(f"🟢 Model save to '{config_settings.model_save_path}'.")