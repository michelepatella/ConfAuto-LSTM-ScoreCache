from collections import defaultdict
import torch
from utils.feedforward_utils import _compute_forward
from utils.log_utils import info, debug


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
    info("🔄 Batch inference started...")

    # debugging
    debug(f"⚙️ Input loader batch size: {len(loader)}.")

    # initialize data
    total_loss = 0.0
    all_preds, all_targets, all_outputs = [], [], []
    loss_per_class = defaultdict(list)

    model.eval()

    try:
        with torch.no_grad():
            for x_features, x_keys, y_key in loader:

                # debugging
                debug(f"⚙️ Batch x_features shape: {x_features.shape}.")
                debug(f"⚙️ Batch x_keys shape: {x_keys.shape}.")
                debug(f"⚙️ Batch y_key shape: {y_key.shape}.")

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
                debug(f"⚙️ Loss computed: {loss}.")
                debug(f"⚙️ Outputs shape: {outputs.shape}.")

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
                        debug(f"⚙️ (Class-Loss): ({int(class_id.item())} - {class_loss.item()}).")

    except (IndexError, IndexError, KeyError, AttributeError, TypeError) as e:
        raise RuntimeError(f"❌ Error while inferring the batch: {e}.")

    # show a successful message
    info("🟢 Batch inferred.")

    return total_loss, loss_per_class, all_preds, all_targets, all_outputs