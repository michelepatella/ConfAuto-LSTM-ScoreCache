import torch


def _evaluate_model(model, loader, criterion, device):
    """
    Evaluate a model on a validation dataset.
    :param model: The model to evaluate.
    :param loader: The validation loader.
    :param criterion: The loss function.
    :param device: Device to use.
    :return: The average loss.
    """
    # evaluate the model
    model.eval()

    # initialize the total loss
    total_loss = 0.0

    # calculate the average loss
    with torch.no_grad():
        for x, y in loader:

            # unpack
            x_keys, x_timestamps, x_features = x
            x_keys = x_keys.to(device)
            x_timestamps = x_timestamps.to(device)
            x_features = x_features.to(device)
            y = y.to(device)

            # calculate the outputs
            outputs = model(x_features, x_timestamps, x_keys)

            # calculate the loss and update the total one
            loss = criterion(outputs, y)
            total_loss += loss.item()

    # return the average loss
    return total_loss / len(loader)