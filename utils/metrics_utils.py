import torch
from sklearn.metrics import classification_report
from utils.log_utils import info, debug


def _top_k_accuracy(
        targets,
        outputs,
        config_settings
):
    """
    To calculate the top-k accuracy of the predictions.
    :param targets: The targets.
    :param outputs: The outputs of the model.
    :param config_settings: The configuration settings.
    :return: The k-accuracy of the predictions.
    """
    try:
        # prepare data
        outputs_tensor = torch.stack(outputs)
        top_k_preds = (torch.topk(
            outputs_tensor,
            k=config_settings.top_k,
            dim=1
        ).indices.cpu().numpy())

        # initialize the no. of correct predictions
        correct = 0

        # count the correct predictions
        for i in range(len(targets)):

            # get the top-k predictions
            top_k_i = top_k_preds[i][:config_settings.top_k]

            # check if the target is contained into the
            # top-k predictions
            if targets[i] in top_k_i:
                correct += 1

        # calculate the accuracy
        accuracy = correct / len(targets)

    except (RuntimeError, IndexError, TypeError, ZeroDivisionError, ValueError) as e:
        raise RuntimeError(f"❌ Error while computing top-k accuracy: {e}.")

    return accuracy


def _compute_metrics(
        targets,
        predictions,
        outputs,
        config_settings
):
    """
    Method to compute metrics based on predictions and targets.
    :param targets: The targets.
    :param predictions: Predictions from model.
    :param outputs: Probabilities from model.
    :param config_settings: The configuration settings.
    :return: The computed metrics.
    """
    # initial message
    info("🔄 Metrics computation started...")

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
            config_settings
        )
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"❌ Error while computing metrics: {e}.")

    # collect metrics
    metrics = {
        "class_metrics": class_report,
        "top_k_accuracy": top_k_accuracy,
    }

    # show a successful message
    info("🟢 Metrics computed.")

    return metrics


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
    info("🔄 Average losses calculation started...")

    # debugging
    debug(f"⚙️ Total loss: {total_loss}.")
    debug(f"⚙️ Number of batches: {num_batches}.")

    try:
        # compute the average loss
        avg_loss = total_loss / num_batches

        # compute the average loss per class
        avg_loss_per_class = {
            cls: sum(losses) / len(losses)
            for cls, losses in loss_per_class.items()
        }
    except (ZeroDivisionError, TypeError, AttributeError, KeyError) as e:
        raise RuntimeError(f"❌ Error while calculating average losses: {e}.")

    # show a successful message
    info("🟢 Average losses calculated.")

    return avg_loss, avg_loss_per_class