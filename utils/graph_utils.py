from matplotlib.colors import LogNorm
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
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()
    except (NameError, AttributeError, TypeError, ValueError, IndexError) as e:
        raise RuntimeError(f"❌ Error while building Zipg log-log plot: {e}.")

    # show a successful message
    info("🟢 Zipf log-log plot built.")


def plot_keys_transition_matrix(requests):
    """
    Method to plot the transition matrix of the keys.
    :param requests: The requests generated.
    :return:
    """
    # initial message
    info("🔄 Keys transition matrix building started...")

    try:
        # check requests list
        if not requests:
            raise ValueError("❌ Request list is empty, "
                             "cannot generate plot.")

        # sort the requests
        unique_keys = sorted(set(requests))

        # get index of keys
        key_to_idx = {
            key: i for i,
            key in enumerate(unique_keys)
        }

        # initialize the transition matrix
        matrix = np.zeros((
            len(unique_keys),
            len(unique_keys)
        ))

        # for each request
        for i in range(1, len(requests)):
            # the position of the transition matrix
            # corresponding to i-th key to j-th key is
            # increased by one, indicating moving from
            # i-th key to the j-th one
            from_key = key_to_idx[requests[i - 1]]
            to_key = key_to_idx[requests[i]]
            matrix[from_key][to_key] += 1

        # plot the transition matrix in form of heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            norm=LogNorm(),
            cmap="YlGnBu"
        )
        plt.xlabel("To Key")
        plt.ylabel("From Key")
        plt.title("Key Transition Heatmap")
        plt.tight_layout()
        plt.show()
        plt.close()
    except (NameError, TypeError, SyntaxError, IndexError, KeyError, ValueError) as e:
        raise RuntimeError(f"❌ Error while building the "
                           f"transition matrix of keys: {e}.")

    # show a successful message
    info("🟢 Keys transition matrix built.")


def plot_requests_over_time(requests, timestamps, bin_size=1000):
    """
    Method to plot the requests over the time.
    :param bin_size: The bin size to consider when
    plotting the requests over the time.
    :param requests: The requests generated.
    :param timestamps: The timestamps generated.
    :return:
    """
    # initial message
    info("🔄 Requests over time plot building started...")

    try:
        # check requests and timestamps lists
        if not requests or not timestamps:
            raise ValueError("❌ Request and/or timestamps lists is empty, "
                             "cannot generate plot.")

        # get the max timestamp (the last element)
        max_time = timestamps[-1]

        # calculate the number of bin required
        # to cover max_time
        num_bins = int(max_time // bin_size) + 1


        bins = [0] * num_bins
        for ts in timestamps:
            # for each timestamp, find its bin
            # increasing the occurrences of this latter
            bin_index = int(ts // bin_size)
            bins[bin_index] += 1

        # define the position of each bin
        x_vals = [i * bin_size for i in range(num_bins)]

        # show requests over time
        plt.figure(figsize=(12, 10))
        plt.plot(x_vals, bins, marker='o', linewidth=1)
        plt.xlabel(f"Time (ms), bin size = {bin_size}")
        plt.ylabel("Number of Requests")
        plt.title("Requests Over Time (Binned)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()
    except (TypeError, IndexError, ValueError) as e:
        raise RuntimeError(f"❌ Error while building the requests"
                           f" over time plot: {e}.")

    # show a successful message
    info("🟢 Requests over time plot built.")


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
        plt.grid(True)
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
        plt.grid(True)
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


def plot_key_usage_heatmap(
        requests,
        timestamps,
        config_settings
):
    """
    Plot heatmap of key access frequencies for all 24 hours.
    :param requests: List of keys requested.
    :param timestamps: Corresponding list of timestamps (in seconds).
    :param config_settings: Config settings for key range.
    """
    # show initial message
    info("🔄 Plotting key usage heatmap for all 24 hours started...")

    try:
        day_seconds = 24 * 60 * 60
        num_keys = config_settings.last_key - config_settings.first_key
        heatmap = np.zeros(
            (24, num_keys),
            dtype=int
        )

        # fill heatmap
        for key, ts in zip(requests, timestamps):
            hour = int((ts % day_seconds) / 3600)
            key_idx = key - config_settings.first_key
            if 0 <= hour < 24 and 0 <= key_idx < num_keys:
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
                num_keys,
                max(1, num_keys // 20)
            ),
            labels=np.arange(
                config_settings.first_key,
                config_settings.last_key,
                max(1, num_keys // 20)
            ),
            rotation=90)
        plt.tight_layout()
        plt.show()
        plt.close()
    except (AttributeError, TypeError, ValueError, IndexError) as e:
        raise RuntimeError(f"❌ Error while building the confusion matrix plot: {e}.")

    # show successful message
    info("🟢 Key usage heatmap for all 24 hours plotted.")