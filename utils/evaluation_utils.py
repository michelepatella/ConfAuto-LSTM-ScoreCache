import torch
import numpy as np
from utils.graph_utils import plot_precision_recall_curve, plot_class_report
from utils.inference_utils import _infer_batch
from utils.log_utils import info, debug
from utils.metrics_utils import _compute_metrics, _calculate_average_losses


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
    :return: The average loss, the metrics, the cost, all the outputs
    and the all the variances.
    """
    # initial message
    info("🔄 Model's evaluation started...")

    # infer the batch
    (total_loss, loss_per_class,
     all_preds, all_targets, all_outputs, all_vars) = _infer_batch(
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

    if compute_metrics:
        # show results
        info(f"📉 Average Loss: {avg_loss}")
        info(f"📉 Average Loss per Class:")

        for i, avg in enumerate(avg_loss_per_class):
            info(f"— Key {i + 1}: {avg}")

        info(f"📉 Class Report per Class:")
        info(f"{metrics['class_report']}")

        info(f"\nConfusion Matrix:\n{np.array(metrics['confusion_matrix'])}")

        info(f"📉 Top-k Accuracy: {metrics['top_k_accuracy']}")
        info(f"📉 Kappa Statistic: {metrics['kappa_statistic']}")

        # show some plots
        plot_precision_recall_curve(
            all_targets,
            torch.stack(all_outputs).numpy(),
            config_settings.num_keys
        )
        plot_class_report(metrics["class_report"])

    # show a successful message
    info("🟢 Model's evaluation completed.")

    return (avg_loss, avg_loss_per_class,
            metrics, all_outputs, all_vars)