from utils.log_utils import _info
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter


def _plot_key_frequencies(requests):
    """
    Method to plot the frequency of the keys in the dataset.
    :param requests: Requests created.
    :return:
    """
    # initial message
    _info("🔄 Keys frequency plot building started...")

    try:

        key_counts = Counter(requests)

        keys = sorted(key_counts.keys())
        freqs = [key_counts[k] for k in keys]

        plt.figure(figsize=(10, 5))
        plt.bar(keys, freqs)
        plt.xlabel("Key")
        plt.ylabel("Frequency")
        plt.title("Access Frequency per Key")
        plt.tight_layout()
        plt.close()

    except (TypeError, ValueError, ZeroDivisionError) as e:
        raise RuntimeError(f"❌ Error while building the keys frequency plot: {e}.")

    # show a successful message
    _info("🟢 Keys frequency plot built.")


def plot_inter_arrival_times(delta_times):

    plt.figure(figsize=(10, 5))
    plt.hist(delta_times, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel("Delta Time (s)")
    plt.ylabel("Count")
    plt.title("Distribution of Inter-arrival Times")
    plt.tight_layout()
    plt.close()


def plot_requests_over_time(requests, timestamps):

    plt.figure(figsize=(10, 5))
    plt.scatter(timestamps[:len(requests)], requests, s=10, alpha=0.6)
    plt.xlabel("Timestamp")
    plt.ylabel("Key Requested")
    plt.title("Requests Over Time")
    plt.tight_layout()
    plt.close()


def plot_transition_matrix(requests):

    unique_keys = sorted(set(requests))
    key_to_idx = {key: i for i, key in enumerate(unique_keys)}
    matrix = np.zeros((len(unique_keys), len(unique_keys)))

    for i in range(1, len(requests)):
        from_key = key_to_idx[requests[i - 1]]
        to_key = key_to_idx[requests[i]]
        matrix[from_key][to_key] += 1

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=unique_keys, yticklabels=unique_keys, cmap="YlGnBu")
    plt.xlabel("To Key")
    plt.ylabel("From Key")
    plt.title("Key Transition Heatmap")
    plt.tight_layout()
    plt.close()