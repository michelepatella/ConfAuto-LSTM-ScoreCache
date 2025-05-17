from utils.log_utils import _info, _debug


def _compute_forward(batch, model, criterion, device):
    """
    Method to compute forward pass and loss based on batch of data.
    :param batch: The batch of data to process.
    :param model: The model to use.
    :param criterion: The loss function.
    :param device: The device to use.
    :return: The loss function for the batch.
    """
    # initial message
    _info("🔄 Forward pass started...")

    # try unpack data
    try:
        # unpack data
        x_features, x_keys, y_key = batch

        # debugging
        _debug(f"⚙️ Target batch: {y_key}.")
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"❌ Error while error unpacking data: {e}")

    try:
        # move data to the device
        x_features = x_features.to(device)
        x_keys = x_keys.to(device)
        y_key = y_key.to(device)
    except (AttributeError, TypeError) as e:
        raise RuntimeError(f"❌ Error while moving data to device: {e}")

    try:
        # calculate the outputs
        outputs = model(x_features, x_keys)

        # debugging
        _debug(f"⚙️ Input batch shape: {x_features.shape}.")
        _debug(f"⚙️ Input keys shape: {x_keys.shape}")
        _debug(f"⚙️ Model output shape: {outputs.shape}.")

    except (TypeError, AttributeError) as e:
        raise RuntimeError(f"❌ Error during model inference: {e}")

    try:
        # calculate the loss and update the total one
        loss = criterion(outputs, y_key)

        # debugging
        _debug(f"⚙️ Loss: {loss.item()}.")
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"❌ Error while calculating loss: {e}")

    # show a successful message
    _info("🟢 Forward pass computed.")

    return loss, outputs


def _compute_backward(loss, optimizer):
    """
    Method to compute backward pass.
    :param loss: The loss to back propagate.
    :param optimizer: The optimizer to use.
    :return:
    """
    # initial message
    _info("🔄 Backward pass started...")

    try:
        # backward pass
        loss.backward()

        # optimize backward pass
        optimizer.step()
    except (AttributeError, TypeError) as e:
        raise RuntimeError(f"❌ Error during backpropagation: {e}")

    # show a successful message
    _info("🟢 Backward pass computed.")