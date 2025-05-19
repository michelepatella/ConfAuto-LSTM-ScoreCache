from utils.inference_utils import _infer_batch
from utils.log_utils import _info, _debug
from utils.metrics_utils import _compute_metrics, _calculate_average_losses


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
        device,
        config_settings
):
    """
    Method to orchestrate the model evaluation on a loader.
    :param model: The model to evaluate.
    :param loader: The loader on which to evaluate the model.
    :param criterion: The loss function.
    :param device: Device to use.
    :param config_settings: The configuration settings.
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
        all_outputs,
        config_settings
    )

    # show results
    _info(f"📉 Average Loss: {avg_loss}")
    _info(f"📉 Average Loss per Class: {avg_loss_per_class}")
    _info(f"📊 Metrics: {metrics}")

    # show a successful message
    _info("🟢 Model's evaluation completed.")

    return avg_loss, avg_loss_per_class, metrics