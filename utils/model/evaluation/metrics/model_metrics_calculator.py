from sklearn.metrics import classification_report, confusion_matrix
from utils.logs.log_utils import info
from utils.model.evaluation.metrics.kappa_statistic_calculator import calculate_kappa_statistic
from utils.model.evaluation.metrics.top_k_calculator import calculate_top_k_accuracy


def compute_model_metrics(
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
        top_k_accuracy = calculate_top_k_accuracy(
            targets,
            outputs,
            config_settings
        )

        # compute the confusion matrix
        conf_matrix = confusion_matrix(
            targets,
            predictions
        )

        # calculate kappa statistic
        kappa_statistic = calculate_kappa_statistic(
            targets,
            predictions
        )

    except (
            ValueError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while computing metrics: {e}.")

    # collect metrics
    metrics = {
        "class_report": class_report,
        "top_k_accuracy": top_k_accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "kappa_statistic": kappa_statistic
    }

    # show a successful message
    info("🟢 Metrics computed.")

    return metrics