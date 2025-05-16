from collections import defaultdict
import torch
from sklearn.metrics import classification_report
from utils.log_utils import _info, _debug
from utils.config_utils import _get_config_value
from utils.feedforward_utils import _compute_forward

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
            for x, y in loader:

                # debugging
                _debug(f"⚙️ Batch x shape: {x.shape}.")
                _debug(f"⚙️ Batch y shape: {y.shape}.")

                x = x.to(device)
                y = y.to(device)

                # calculate loss and outputs through forward pass
                loss, outputs = _compute_forward(
                    (x, y),
                    model,
                    criterion,
                    device
                )

                # debugging
                _debug(f"⚙️ Loss computed: {loss}.")
                _debug(f"⚙️ Outputs shape: {outputs.shape}.")

                # check the loss
                if loss is None:
                    raise Exception("❌ Error while computing average loss due to loss equals None.")

                # update the total loss
                total_loss += loss.item()

                # store predictions and target for metrics
                preds = torch.argmax(outputs, dim=1)

                # debugging
                _debug(f"⚙️ Predictions: {preds}.")

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                all_outputs.extend(outputs.cpu())

                # calculate loss per class
                for class_id in torch.unique(y):
                    mask = y == class_id
                    if mask.sum() > 0:
                        class_loss = criterion(outputs[mask], y[mask])
                        loss_per_class[int(class_id.item())].append(class_loss.item())

                        # debugging
                        _debug(f"⚙️ (Class-Loss): ({int(class_id.item())} - {class_loss.item()}).")

    except Exception as e:
        raise Exception(f"❌ Error while inferring the batch: {e}")

    # show a successful message
    _info("🟢 Batch inferred.")

    return total_loss, loss_per_class, all_preds, all_targets, all_outputs


def _calculate_average_losses(
        total_loss,
        loss_per_class,
        num_batches
):
    """
    Method to calculate average losses (global and per class).
    :param total_loss: The total global loss.
    :param loss_per_class: The total loss per class.
    :param num_batches: The number of batches.
    :return: The average global loss and the average loss per class.
    """
    # initial message
    _info("🔄 Average losses calculation started...")

    # debugging
    _debug(f"⚙️ Total loss: {total_loss}.")
    _debug(f"⚙️ Number of batches: {num_batches}.")

    try:
        # compute the average loss
        avg_loss = total_loss / num_batches

        # compute the average loss per class
        avg_loss_per_class = {
            cls: sum(losses) / len(losses)
            for cls, losses in loss_per_class.items()
        }

    except Exception as e:
        raise Exception(f"❌ Error while calculating average losses: {e}")

    # show a successful message
    _info("🟢 Average losses calculated.")

    return avg_loss, avg_loss_per_class


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
        raise Exception("❌ Error while collecting predictions due to empty loader.")

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
    _debug(f"⚙️ Predictions: {all_preds}.")
    _debug(f"⚙️ Targets: {all_targets}.")

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


def _top_k_accuracy(targets, outputs, k):
    """
    To calculate the top-k accuracy of the predictions.
    :param targets: The targets.
    :param outputs: The outputs of the model.
    :param k: The value of k for the accuracy.
    :return: The k-accuracy of the predictions.
    """
    # prepare data
    outputs_tensor = torch.stack(outputs)
    top_k_preds = (torch.topk(outputs_tensor, k=k, dim=1)
                   .indices.cpu().numpy())

    # initialize the no. of correct predictions
    correct = 0

    # count the correct predictions
    for i in range(len(targets)):

        # get the top-k predictions
        top_k_i = top_k_preds[i][:k]

        # check if the target is contained into the
        # top-k predictions
        if targets[i] in top_k_i:
            correct += 1

    return correct / len(targets)


def _compute_metrics(targets, predictions, outputs):
    """
    Method to compute metrics based on predictions and targets.
    :param targets: The targets.
    :param predictions: Predictions from model.
    :param outputs: Probabilities from model.
    :return: The computed metrics.
    """
    # initial message
    _info("🔄 Metrics computation started...")

    # load some configurations
    top_k = _get_config_value("evaluation.top_k")

    try:
        # class-wise metrics
        class_report = classification_report(
            targets,
            predictions,
            output_dict=True,
            zero_division=0
        )

        # calculate the top-k accuracy
        top_k_accuracy = _top_k_accuracy(
            targets,
            outputs,
            top_k
        )

    except Exception as e:
        raise Exception(f"❌ Error while computing metrics: {e}")

    # collect metrics
    metrics = {
        "class_metrics": class_report,
        "top_k_accuracy": top_k_accuracy,
    }

    # debugging
    _debug(f"⚙️ Classification report keys: {list(class_report.keys())}.")
    _debug(f"⚙️ Top-k accuracy: {top_k_accuracy}.")

    # show a successful message
    _info("🟢 Metrics computed.")

    return metrics


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