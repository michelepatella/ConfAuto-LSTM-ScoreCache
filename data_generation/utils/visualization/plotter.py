from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from utils.logs.log_utils import info


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
        plt.title(
            "Zipf Distribution (Log-Log)",
            fontsize=18
        )
        plt.xlabel(
            "Key",
            fontsize=16
        )
        plt.ylabel(
            "Frequency",
            fontsize=16
        )
        plt.tight_layout()
        plt.show()
        plt.close()
    except (
            NameError,
            AttributeError,
            TypeError,
            ValueError,
            IndexError
    ) as e:
        raise RuntimeError(f"❌ Error while building Zipg log-log plot: {e}.")

    # show a successful message
    info("🟢 Zipf log-log plot built.")


def plot_daily_profile(
        timestamps,
        bin_size=0.5
):
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
        bins = np.linspace(
            0,
            24,
            num_bins + 1
        )

        # define the histogram
        counts, _ = np.histogram(
            hours,
            bins=bins
        )

        plt.figure(figsize=(12, 10))
        plt.bar(
            bins[:-1],
            counts,
            width=bin_size,
            align='edge',
            edgecolor='black'
        )
        plt.xlabel(
            "Hour of Day",
            fontsize=16
        )
        plt.ylabel(
            "Number of Requests",
            fontsize=16
        )
        plt.title(
            "Distribution of Requests Over 24 Hours",
            fontsize=18
        )
        plt.xticks(
            np.arange(0, 25, step=1),
            fontsize=16
        )
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
        for key, ts in zip(
                requests,
                timestamps
        ):
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
        plt.xlabel(
            'Key',
            fontsize=16
        )
        plt.ylabel(
            'Hour of Day',
            fontsize=16
        )
        plt.title(
            'Heatmap of Key Access Frequency by Hour of Day',
            fontsize=18
        )
        plt.yticks(
            ticks=np.arange(24),
            labels=[f"{h}:00" for h in range(24)],
            fontsize=16
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
            rotation=90,
            fontsize=16
        )
        plt.tight_layout()
        plt.show()
        plt.close()
    except (
            AttributeError,
            TypeError,
            ValueError,
            IndexError
    ) as e:
        raise RuntimeError(f"❌ Error while building the confusion matrix plot: {e}.")

    # show successful message
    info("🟢 Key usage heatmap for all 24 hours plotted.")