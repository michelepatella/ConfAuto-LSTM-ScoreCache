import torch
from utils.logs.log_utils import info, debug


def enable_mc_dropout(model):
    """
    Method to enable MC dropout.
    :param model: The model for which to enable MC dropout.
    :return:
    """
    # initial message
    info("🔄 MC dropout enabling started...")

    try:
        for module in model.modules():
            if isinstance(
                    module,
                    torch.nn.Dropout
            ):
                module.train()
                # debugging
                debug(f"⚙️ Dropout enabled.")

        # set dropout enabled
        model.use_mc_dropout = True
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 MC dropout enabled.")