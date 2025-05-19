from matplotlib.colors import LogNorm
from sklearn.metrics import precision_recall_curve, average_precision_score
from utils.log_utils import info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import label_binarize


def plot_key_frequencies_histogram(requests):
    """
    Method to plot the frequency of key accesses via histogram.
    :param requests: The requests generated.
    :return:
    """
    # initial message
    info("🔄 Key frequencies histogram building started...")

    try:
        # check requests list
        if not requests:
            raise ValueError("❌ Request list is empty, cannot generate plot.")

        # count the keys
        key_counts = Counter(requests)

        # sort the keys
        keys = sorted(key_counts.keys())

        # count the frequency of each key
        freqs = [key_counts[k] for k in keys]

        # plot the histogram showing the
        # frequency key accesses
        plt.figure(figsize=(10, 5))
        plt.bar(keys, freqs)
        plt.xlabel("Key")
        plt.ylabel("Frequency")
        plt.title("Access Frequency per Key")
        plt.tight_layout()
        plt.show()
        plt.close()
    except (NameError, TypeError, ImportError, AttributeError, KeyError) as e:
        raise RuntimeError(f"❌ Error while building key frequencies histogram: {e}.")

    # show a successful message
    info("🟢 Key frequencies histogram built.")


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


def plot_requests_over_time(requests, delta_times):
    """
    Method to plot the requests over the time.
    :param requests: The requests generated.
    :param delta_times: The delta times generated.
    :return:
    """
    # initial message
    info("🔄 Requests over time plot building started...")

    try:
        # check requests and delta times lists
        if not requests or not delta_times:
            raise ValueError("❌ Request and/or delta times lists is empty, "
                             "cannot generate plot.")

        # construct timestamps starting from delta times
        timestamps = [0]
        for dt in delta_times:
            timestamps.append(timestamps[-1] + dt)
        timestamps = timestamps[1:]

        # show requests over time
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, requests)
        plt.ylabel("Key Requested")
        plt.xlabel("Timestamps")
        plt.title("Requests Over Time")
        plt.grid(True)
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
        targets_bin = label_binarize(
            targets,
            classes=np.arange(num_keys)
        )

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
        plt.legend()
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
            plot(kind='bar', figsize=(12, 6))
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