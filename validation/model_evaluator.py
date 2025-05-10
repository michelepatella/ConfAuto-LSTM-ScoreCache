import logging
import torch
from validation.batch_processor import _process_batch


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

    # check the loader dimension
    if len(loader) == 0:
        logging.warning("The loader is empty. Skipping evaluation.")
        return 0.0

    # calculate the average loss
    with torch.no_grad():
        for x, y in loader:

            # calculate loss by processing the batch
            loss = _process_batch((x, y), model, criterion, device)

            # check loss
            if loss is None:
                return None

            # update total loss
            total_loss += loss.item()

    # try to calculate the average loss
    try:
        # calculate the average loss
        avg_loss = total_loss / len(loader)
    except ZeroDivisionError:
        logging.error("ZeroDivisionError: The loader has no batches.")
        return None

    return avg_loss