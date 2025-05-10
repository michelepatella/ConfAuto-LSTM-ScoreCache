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
        x_features, y_key = batch
    except Exception as e:
        raise Exception(f"An unexpected error while error unpacking data: {e}")

    # try to move data to the device
    try:
        # move data to the device
        x_features = x_features.to(device)
        y_key = y_key.to(device)
    except Exception as e:
        raise Exception(f"An unexpected error while moving data to device: {e}")

    # try to calculate the outputs
    try:
        # calculate the outputs
        outputs = model(x_features)
    except Exception as e:
        raise Exception(f"An unexpected error during model inference: {e}")

    # try to calculate the loss and do the update
    try:
        # calculate the loss and update the total one
        loss = criterion(outputs, y_key)
    except Exception as e:
        raise Exception(f"An unexpected error while calculating loss: {e}")

    return loss