import numpy as np
from utils.inference_utils import _infer_batch
from utils.log_utils import info, debug
from utils.metrics_utils import _compute_model_standalone_metrics


def evaluate_model(
        model,
        loader,
        criterion,
        device,
        config_settings,
        compute_metrics=False
):
    """
    Method to orchestrate the model evaluation on a loader.
    :param model: The model to evaluate.
    :param loader: The loader on which to evaluate the model.
    :param criterion: The loss function.
    :param device: Device to use.
    :param config_settings: The configuration settings.
    :param compute_metrics: Whether to compute metrics or not.
    :return: The average loss, the metrics, all the outputs,
    all the targets, and the all the variances.
    """
    # initial message
    info("🔄 Model's evaluation started...")

    # infer the batch
    (
        total_loss,
        all_preds,
        all_targets,
        all_outputs,
        all_vars
    ) = _infer_batch(
        model,
        loader,
        criterion,
        device,
        config_settings
    )

    # debugging
    debug(f"⚙️ Total predictions collected: {len(all_preds)}.")
    debug(f"⚙️ Total targets: {len(all_targets)}.")

    # calculate the average of losses
    avg_loss = total_loss / len(loader)

    metrics = None
    if compute_metrics:
        # compute metrics
        metrics = _compute_model_standalone_metrics(
            all_targets,
            all_preds,
            all_outputs,
            config_settings
        )

        # show results
        info(f"📉 Average Loss: {avg_loss}")

        info(f"📉 Class Report per Class:")
        info(f"{metrics['class_report']}")

        info(f"\nConfusion Matrix:\n{np.array(metrics['confusion_matrix'])}")

        info(f"📉 Top-k Accuracy: {metrics['top_k_accuracy']}")
        info(f"📉 Kappa Statistic: {metrics['kappa_statistic']}")

    # show a successful message
    info("🟢 Model's evaluation completed.")

    return (
        avg_loss,
        metrics,
        all_outputs,
        all_targets,
        all_vars
    )