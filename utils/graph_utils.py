from sklearn.metrics import precision_recall_curve, average_precision_score
from utils.log_utils import info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import label_binarize


def plot_zipf_loglog(requests):
    """
    Method to plot the frequency of key accesses via loglog.
    :param requests: The requests generated.
    :return:
    """
    # initial message
    info("🔄 Zipf log-log plot building started...")

    try:
        # check requests list
        if not requests:
            raise ValueError("❌ Request list is empty, cannot generate plot.")

        # count the requests
        key_counts = Counter(requests)

        # get frequencies
        frequencies = np.array(sorted(
            key_counts.values(),
            reverse=True)
        )

        # get the frequencies for each key
        ranks = np.arange(
            1,
            len(frequencies) + 1
        )

        # plot the zipf distribution loglog plot
        plt.figure()
        plt.loglog(
            ranks,
            frequencies,
            marker='o'
        )
        plt.title("Zipf Distribution (Log-Log)")
        plt.xlabel("Key")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        plt.close()
    except (NameError, AttributeError, TypeError, ValueError, IndexError) as e:
        raise RuntimeError(f"❌ Error while building Zipg log-log plot: {e}.")

    # show a successful message
    info("🟢 Zipf log-log plot built.")


def plot_daily_profile(timestamps, bin_size=0.5):
    """
    Method to plot the distribution of requests over a 24-hour day.
    :param timestamps: Timestamps (in hours).
    :param bin_size: Bin size (in hours).
    """
    # show initial message
    info("🔄 Daily request profile plot building started...")

    try:
        # extract hours
        hours = np.array(timestamps) % 24

        # define bins
        num_bins = int(24 / bin_size)
        bins = np.linspace(0, 24, num_bins + 1)

        # define the histogram
        counts, _ = np.histogram(hours, bins=bins)

        plt.figure(figsize=(12, 10))
        plt.bar(
            bins[:-1],
            counts,
            width=bin_size,
            align='edge',
            edgecolor='black'
        )
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Requests")
        plt.title("Distribution of Requests Over 24 Hours")
        plt.xticks(np.arange(0, 25, step=1))
        plt.tight_layout()
        plt.show()
        plt.close()

    except Exception as e:
        raise RuntimeError(f"❌ Error while building daily request profile: {e}.")

    info("🟢 Daily request profile plot built.")


def plot_key_usage_heatmap(
        requests,
        timestamps,
        config_settings
):
    """
    Method to plot the heatmap of key access frequencies for all 24 hours.
    :param requests: List of keys.
    :param timestamps: List of timestamps.
    :param config_settings: Config settings.
    """
    # show initial message
    info("🔄 Plotting key usage heatmap for all 24 hours started...")

    try:
        # define the heatmap
        heatmap = np.zeros(
            (24, config_settings.num_keys),
            dtype=int
        )

        # fill the heatmap
        for key, ts in zip(requests, timestamps):
            hour = int(ts)
            key_idx = key - config_settings.first_key
            if (
                0 <= hour < 24 and
                0 <= key_idx < config_settings.num_keys
            ):
                heatmap[hour, key_idx] += 1

        plt.figure(figsize=(12, 10))
        plt.imshow(
            heatmap,
            aspect='auto',
            cmap='viridis'
        )
        plt.colorbar(label='Access Count')
        plt.xlabel('Key')
        plt.ylabel('Hour of Day')
        plt.title('Heatmap of Key Access Frequency by Hour of Day')
        plt.yticks(
            ticks=np.arange(24),
            labels=[f"{h}:00" for h in range(24)]
        )
        plt.xticks(
            ticks=np.arange(
                0,
                config_settings.num_keys,
                max(1, config_settings.num_keys // 20)
            ),
            labels=np.arange(
                config_settings.first_key,
                config_settings.last_key,
                max(1, config_settings.num_keys // 20)
            ),
            rotation=90)
        plt.tight_layout()
        plt.show()
        plt.close()
    except (AttributeError, TypeError, ValueError, IndexError) as e:
        raise RuntimeError(f"❌ Error while building the confusion matrix plot: {e}.")

    # show successful message
    info("🟢 Key usage heatmap for all 24 hours plotted.")


def plot_precision_recall_curve(targets, outputs, num_keys):
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

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.show()
        plt.close()

    except (ValueError, TypeError, IndexError) as e:
        raise RuntimeError(f"❌ Error while building the precision-recall curve: {e}.")

    # show a successful message
    info("🟢 Precision-Recall curve built.")


def plot_class_report(class_report):
    """
    Method to plot the class report.
    :param class_report: The computed class report.
    :return:
    """
    # initial message
    info("🔄 Class report plot building started...")

    try:
        # extract precision, recall, and f1-score
        class_report_filtered = {
            k: v for k, v in class_report.items()
            if k.isdigit() or isinstance(k, int)
        }

        # transform to dataframe
        df = pd.DataFrame(class_report_filtered).T

        (df[['precision', 'recall', 'f1-score']].
            plot(kind='bar', figsize=(12, 10))
         )
        plt.title("Classification Report Metrics per Class")
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.show()
        plt.close()

        # show a successful message
        info("🟢 Class report plot built.")
    except (AttributeError, TypeError, ValueError, KeyError) as e:
        raise RuntimeError(f"❌ Error while building the class report plot: {e}.")


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
        plt.title("Confusion Matrix")
        plt.ylabel("True Key")
        plt.xlabel("Predicted Key")
        plt.tight_layout()
        plt.show()
        plt.close()
    except (AttributeError, TypeError, ValueError) as e:
        raise RuntimeError(f"❌ Error while building the confusion matrix plot: {e}.")

    # show a successful message
    info("🟢 Confusion matrix built.")