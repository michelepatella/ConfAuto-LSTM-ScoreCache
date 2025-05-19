from collections import Counter

import torch
from sklearn.metrics import classification_report
from utils.log_utils import _info, _debug


def _top_k_accuracy(targets, outputs):
    """
    To calculate the top-k accuracy of the predictions.
    :param targets: The targets.
    :param outputs: The outputs of the model.
    :return: The k-accuracy of the predictions.
    """
    from main import config_settings

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
            outputs
        )
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"❌ Error while computing metrics: {e}.")

    # collect metrics
    metrics = {
        "class_metrics": class_report,
        "top_k_accuracy": top_k_accuracy,
    }

    # show a successful message
    _info("🟢 Metrics computed.")

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
    except (ZeroDivisionError, TypeError, AttributeError, KeyError) as e:
        raise RuntimeError(f"❌ Error while calculating average losses: {e}.")

    # show a successful message
    _info("🟢 Average losses calculated.")

    return avg_loss, avg_loss_per_class


def _calculate_rel_frequency(sequence, window):
    """
    Method to calculate the relative frequency of a specific sequence in a given window.
    :param sequence: The sequence to calculate the frequency of.
    :param window: The window within to calculate the frequency.
    :return: The frequency of the sequence.
    """
    # initial message
    _info("🔄 Relative frequency sequence calculation started...")

    # debugging
    _debug(f"⚙️ Sequence length: {len(sequence)}.")
    _debug(f"⚙️ Window: {window}.")

    # initialize frequencies
    freqs = []

    try:
        # count the frequency of the sequence
        # within the given temporal window
        for i in range(len(sequence)):
            if i < window:
                recent = sequence[:i]
            else:
                recent = sequence[i - window:i]

            # calculate the relative frequency
            count = Counter(recent)
            freq = count[sequence[i]] / len(recent) \
                if len(recent) > 0 \
                else 0.0
            freqs.append(freq)
    except (TypeError, ZeroDivisionError,
            IndexError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while calculating relative frequency sequence: {e}.")

    # debugging
    _debug(f"⚙️ Frequencies length: {len(freqs)}.")

    # show a successful message
    _info(f"🟢 Relative frequency of the sequence calculated.")

    return freqs
