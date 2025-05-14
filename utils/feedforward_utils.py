import logging


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
    logging.info("🔄 Forward pass started...")

    # try unpack data
    try:
        # unpack data
        x_features, y_key = batch
    except Exception as e:
        raise Exception(f"❌ Error while error unpacking data: {e}")

    try:
        # move data to the device
        x_features = x_features.to(device)
        y_key = y_key.to(device)
    except Exception as e:
        raise Exception(f"❌ Error while moving data to device: {e}")

    try:
        # calculate the outputs
        outputs = model(x_features)
    except Exception as e:
        raise Exception(f"❌ Error during model inference: {e}")

    try:
        # calculate the loss and update the total one
        loss = criterion(outputs, y_key)
    except Exception as e:
        raise Exception(f"❌ Error while calculating loss: {e}")

    # show a successful message
    logging.info("🟢 Forward pass computed.")

    return loss, outputs


def _compute_backward(loss, optimizer):
    """
    Method to compute backward pass.
    :param loss: The loss to back propagate.
    :param optimizer: The optimizer to use.
    :return:
    """
    # initial message
    logging.info("🔄 Backward pass started...")

    try:
        # backward pass
        loss.backward()
    except Exception as e:
        raise Exception(f"❌ Error during backpropagation: {e}")

    try:
        # optimize backward pass
        optimizer.step()
    except Exception as e:
        raise Exception(f"❌ Error while optimizing: {e}")

    # show a successful message
    logging.info("🟢 Backward pass computed.")