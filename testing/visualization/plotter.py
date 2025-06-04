import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from utils.logs.log_utils import info


def plot_precision_recall_curve(
        targets,
        outputs,
        num_keys
):
    """
    Method to plot the precision and recall curve.
    :param targets: The targets.
    :param outputs: The outputs of the model.
    :param num_keys: The total number of keys.
    :return:
    """
    # initial message
    info("🔄 Precision-Recall curve building started...")

    try:
        # get the one-hot version of the outputs
        targets_bin = np.array(label_binarize(
            targets,
            classes=np.arange(num_keys)
        ))

        outputs = np.array(outputs)

        # precision-recall curve one vs the rest
        for i in range(num_keys):
            precision, recall, _ = precision_recall_curve(
                targets_bin[:, i],
                outputs[:, i]
            )
            avg_precision = average_precision_score(
                targets_bin[:, i],
                outputs[:, i]
            )

            # plot the curve
            plt.plot(
                recall,
                precision,
                label=f"Class {i} (AP = {avg_precision:.2f})"
            )

        plt.xlabel(
            "Recall",
            fontsize=16
        )
        plt.ylabel(
            "Precision",
            fontsize=16
        )
        plt.ylabel(
            "Precision",
            fontsize=16
        )
        plt.title(
            "Precision-Recall Curve",
            fontsize=18
        )
        plt.show()
        plt.close()

    except (
            ValueError,
            TypeError,
            IndexError
    ) as e:
        raise RuntimeError(f"❌ Error while building the precision-recall curve: {e}.")

    # show a successful message
    info("🟢 Precision-Recall curve built.")


def plot_confusion_matrix(confusion_matrix):
    """
    Method to plot the confusion matrix.
    :param confusion_matrix: The computed confusion matrix.
    :return:
    """
    # initial message
    info("🔄 Confusion matrix plot building started...")

    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
        )
        plt.title(
            "Confusion Matrix",
            fontsize=18
        )
        plt.ylabel(
            "True Key",
            fontsize=16
        )
        plt.xlabel(
            "Predicted Key",
            fontsize=16
        )
        plt.tight_layout()
        plt.show()
        plt.close()
    except (
            AttributeError,
            TypeError,
            ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while building the confusion matrix plot: {e}.")

    # show a successful message
    info("🟢 Confusion matrix built.")