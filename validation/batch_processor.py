import logging


def _process_batch(batch, model, criterion, device):
    """
    Method to proces a batch of data (forward pass including loss calculation).
    :param batch: The batch of data to process.
    :param model: The model to use.
    :param criterion: The loss function.
    :param device: The device to use.
    :return: The loss function for the batch as output.
    """
    # try unpack data
    try:
        # unpack data
        x, y = batch
        x_keys, x_timestamps, x_features = x
    except Exception as e:
        logging.error(f"An unexpected error while error unpacking data: {e}")
        return None

    # try to move data to the device
    try:
        # move data to the device
        x_keys = x_keys.to(device)
        x_timestamps = x_timestamps.to(device)
        x_features = x_features.to(device)
        y = y.to(device)
    except Exception as e:
        logging.error(f"An unexpected error while moving data to device: {e}")
        return None

    # try to calculate the outputs
    try:
        # calculate the outputs
        outputs = model(x_features, x_timestamps, x_keys)
    except Exception as e:
        logging.error(f"An unexpected error during model inference: {e}")
        return None

    # try to calculate the loss and do the update
    try:
        # calculate the loss and update the total one
        loss = criterion(outputs, y)
    except Exception as e:
        logging.error(f"An unexpected error while calculating loss: {e}")
        return None

    return loss