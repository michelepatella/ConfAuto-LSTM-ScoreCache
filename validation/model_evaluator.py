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

    # save the current state of use_embedding
    original_use_embedding = getattr(model, 'use_embedding', False)

    # set use_embedding to false
    model.use_embedding = False

    # initialize the total loss
    total_loss = 0.0

    # check the loader dimension
    if len(loader) == 0:
        logging.warning("The loader is empty. Skipping evaluation.")
        # restore the previous state of use_embedding
        model.use_embedding = original_use_embedding
        return 0.0

    try:
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

            # calculate the average loss
            avg_loss = total_loss / len(loader)

    except Exception as e:
        raise Exception(f"An unexpected error while evaluating model: {e}")
    finally:
        # restore the previous state of use_embedding
        model.use_embedding = original_use_embedding

    return avg_loss