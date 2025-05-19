from utils.inference_utils import _infer_batch
from utils.log_utils import info, debug
from utils.metrics_utils import _compute_metrics, _calculate_average_losses


def evaluate_model(
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
    info("🔄 Model's evaluation started...")

    # infer the batch
    (total_loss, loss_per_class,
     all_preds, all_targets, all_outputs) = _infer_batch(
        model,
        loader,
        criterion,
        device
    )

    # debugging
    debug(f"⚙️ Total predictions collected: {len(all_preds)}.")
    debug(f"⚙️ Total targets: {len(all_targets)}.")

    # calculate the average of losses
    avg_loss, avg_loss_per_class = (
        _calculate_average_losses(
            total_loss,
            loss_per_class,
            len(loader)
        )
    )

    # compute metrics
    metrics = _compute_metrics(
        all_targets,
        all_preds,
        all_outputs,
        config_settings
    )

    # show results
    info(f"📉 Average Loss: {avg_loss}")
    info(f"📉 Average Loss per Class: {avg_loss_per_class}")
    info(f"📊 Metrics: {metrics}")

    # show a successful message
    info("🟢 Model's evaluation completed.")

    return avg_loss, avg_loss_per_class, metrics