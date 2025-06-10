from utils.logs.log_utils import info, debug


def compute_forward(
        batch,
        model,
        criterion,
        device
):
    """
    Method to compute forward pass and loss based on batch of data.
    :param batch: The batch of data to process.
    :param model: The model to use.
    :param criterion: The loss function.
    :param device: The device to use.
    :return: The loss function (if computed) and the outputs for the batch.
    """
    # initial message
    info("🔄 Forward pass started...")

    # try unpack data
    try:
        # unpack data
        x_features, x_keys, y_key = batch

        # debugging
        debug(f"⚙️ Target batch: {y_key}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    try:
        # move data to the device
        x_features = x_features.to(device)
        x_keys = x_keys.to(device)
        y_key = y_key.to(device)
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    try:
        # calculate the outputs
        outputs = model(
            x_features,
            x_keys
        )

        # debugging
        debug(f"⚙️ Input batch shape: {x_features.shape}.")
        debug(f"⚙️ Input keys shape: {x_keys.shape}")
        debug(f"⚙️ Model output shape: {outputs.shape}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    loss = None
    if criterion is not None:
        try:
            # calculate the loss and update the total one
            loss = criterion(
                outputs,
                y_key
            )

            # debugging
            debug(f"⚙️ Loss: {loss.item()}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except ValueError as e:
            raise ValueError(f"ValueError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Forward pass computed.")

    return loss, outputs