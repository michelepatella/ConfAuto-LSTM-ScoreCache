import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.feedfarward_utils import _compute_forward


def _compute_avg_loss_and_predictions(
        model,
        loader,
        criterion,
        device,
        compute_metrics
):
    """
    Method to compute the model evaluation.
    :param model: The model to be evaluated.
    :param loader: The data loader.
    :param criterion: The loss function.
    :param device: The device to be used.
    :param compute_metrics: Specifies whether to compute metrics or not.
    :return: The average loss and metrics (if requested).
    """
    model.eval()

    # initialize data
    total_loss = 0.0
    all_preds, all_targets = [], []

    # check the length of the loader
    if len(loader) == 0:
        raise Exception("Error while computing average loss due to empty loader.")

    try:
        with torch.no_grad():
            for x, y in loader:

                # calculate loss and outputs through forward pass
                loss, outputs = _compute_forward((x, y), model, criterion, device)

                # check the loss
                if loss is None:
                    raise Exception("Error while computing average loss due to loss equals None.")

                # update the total loss
                total_loss += loss.item()

                # store predictions and target for metrics (if needed)
                if compute_metrics:
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

    except Exception as e:
        raise Exception(f"An unexpected error while evaluating the model: {e}")

    # compute the average loss
    avg_loss = total_loss / len(loader)

    return avg_loss, all_preds, all_targets

def _compute_metrics(predictions, targets):
    """
    Method to compute metrics based on predictions and targets.
    :param predictions: Predictions from model.
    :param targets: Targets.
    :return: The computed metrics.
    """
    try:
        # compute the metrics
        precision = precision_score(targets, predictions, average="macro")
        recall = recall_score(targets, predictions, average="macro")
        f1 = f1_score(targets, predictions, average="macro")
    except Exception as e:
        raise Exception(f"An unexpected error while computing metrics: {e}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def _evaluate_model(
        model,
        loader,
        criterion,
        device,
        compute_metrics=False
):
    """
    Method to orchestrate the model evaluation on a dataset.
    :param model: The model to evaluate.
    :param loader: The loader on which to evaluate the model.
    :param criterion: The loss function.
    :param device: Device to use.
    :param compute_metrics: Specifies whether to compute metrics or not.
    :return: The average loss.
    """
    try:
        # perform the model evaluation
        avg_loss, all_preds, all_targets = _compute_avg_loss_and_predictions(
            model,
            loader,
            criterion,
            device,
            compute_metrics
        )
    except Exception as e:
        raise Exception(f"An unexpected error while evaluating model: {e}")

    # if metrics are requested, compute them
    if compute_metrics:
        metrics = _compute_metrics(all_preds, all_targets)
        return avg_loss, metrics

    return avg_loss