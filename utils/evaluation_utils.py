from collections import defaultdict
import torch
from utils.log_utils import _info, _debug
from utils.feedforward_utils import _compute_forward
from utils.metrics_utils import _compute_metrics, _calculate_average_losses


def _infer_batch(
        model,
        loader,
        criterion,
        device
):
    """
    Method to infer the batch.
    :param model: The model to be used.
    :param loader: The dataloader.
    :param criterion: The loss function.
    :param device: The device to be used.
    :return: The total loss, the loss per class, all the predictions,
     all the targets, and all the outputs returned by the model.
    """
    # initial message
    _info("🔄 Batch inference started...")

    # debugging
    _debug(f"⚙️ Input loader batch size: {len(loader)}.")

    # initialize data
    total_loss = 0.0
    all_preds, all_targets, all_outputs = [], [], []
    loss_per_class = defaultdict(list)

    model.eval()

    try:
        with torch.no_grad():
            for x_features, x_keys, y_key in loader:

                # debugging
                _debug(f"⚙️ Batch x_features shape: {x_features.shape}.")
                _debug(f"⚙️ Batch x_keys shape: {x_keys.shape}.")
                _debug(f"⚙️ Batch y_key shape: {y_key.shape}.")

                x_features = x_features.to(device)
                y_key = y_key.to(device)

                # calculate loss and outputs through forward pass
                loss, outputs = _compute_forward(
                    (x_features, x_keys, y_key),
                    model,
                    criterion,
                    device
                )

                # debugging
                _debug(f"⚙️ Loss computed: {loss}.")
                _debug(f"⚙️ Outputs shape: {outputs.shape}.")

                # check the loss
                if loss is None:
                    raise ValueError("❌ Error while computing average loss due to loss equals None.")

                # update the total loss
                total_loss += loss.item()

                # store predictions and target for metrics
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_key.cpu().numpy())
                all_outputs.extend(outputs.cpu())

                # calculate loss per class
                for class_id in torch.unique(y_key):  # e qui
                    mask = y_key == class_id
                    if mask.sum() > 0:
                        class_loss = criterion(outputs[mask], y_key[mask])
                        loss_per_class[int(class_id.item())].append(class_loss.item())

                        # debugging
                        _debug(f"⚙️ (Class-Loss): ({int(class_id.item())} - {class_loss.item()}).")

    except (IndexError, IndexError, KeyError, AttributeError, TypeError) as e:
        raise RuntimeError(f"❌ Error while inferring the batch: {e}.")

    # show a successful message
    _info("🟢 Batch inferred.")

    return total_loss, loss_per_class, all_preds, all_targets, all_outputs


def _collect_predictions(
        model,
        loader,
        criterion,
        device
):
    """
    Method to collect predictions and get related information.
    :param model: The model from which prediction will be collected.
    :param loader: The data loader.
    :param criterion: The loss function.
    :param device: The device to be used.
    :return: The predictions (along with the outputs), the targets
    and the both the global and class average loss.
    """
    # initial message
    _info("🔄 Prediction collection started...")

    # check the length of the loader
    if len(loader) == 0:
        raise ValueError("❌ Error while collecting predictions due to empty loader.")

    # infer the batch
    (total_loss, loss_per_class,
    all_preds, all_targets, all_outputs) = _infer_batch(
        model,
        loader,
        criterion,
        device
    )

    # debugging
    _debug(f"⚙️ Total predictions collected: {len(all_preds)}.")
    _debug(f"⚙️ Total targets: {len(all_targets)}.")

    # calculate the average of losses
    avg_loss, avg_loss_per_class = (
        _calculate_average_losses(
            total_loss,
            loss_per_class,
            len(loader)
        )
    )

    # show a successful message
    _info("🟢 Predictions collected.")

    return avg_loss, avg_loss_per_class, all_preds, all_targets, all_outputs


def _evaluate_model(
        model,
        loader,
        criterion,
        device
):
    """
    Method to orchestrate the model evaluation on a loader.
    :param model: The model to evaluate.
    :param loader: The loader on which to evaluate the model.
    :param criterion: The loss function.
    :param device: Device to use.
    :return: The average loss and the metrics.
    """
    # initial message
    _info("🔄 Model's evaluation started...")

    # collect predictions to get them along with
    # targets and average loss
    avg_loss, avg_loss_per_class, all_preds, all_targets, all_outputs = (
        _collect_predictions(
            model,
            loader,
            criterion,
            device
        ))

    # compute metrics
    metrics = _compute_metrics(
        all_targets,
        all_preds,
        all_outputs
    )

    # show results
    _info(f"📉 Average Loss: {avg_loss}")
    _info(f"📉 Average Loss per Class: {avg_loss_per_class}")
    _info(f"📊 Metrics: {metrics}")

    # show a successful message
    _info("🟢 Model's evaluation completed.")

    return avg_loss, avg_loss_per_class, metrics