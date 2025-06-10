from utils.logs.log_utils import info
import matplotlib.pyplot as plt


def plot_hit_miss_rate_over_time(results):
    """
    Method to plot the hit rate and the miss rate over the time.
    :param results: The timeline results.
    :return:
    """
    # initial message
    info("🔄 Hit rate and miss rate plot building started...")

    try:
        plt.figure(figsize=(12, 10))

        # first subplot: hit rate
        plt.subplot(2, 1, 1)
        for result in results:
            # extract hit rate results
            policy = result['policy']
            timeline = result['timeline']
            x = [
                point['index']
                for point in timeline
            ]
            y_hit = [
                point['instant_hit_rate']
                for point in timeline
            ]
            plt.plot(
                x,
                y_hit,
                label=f"{policy} (hit)",
                linestyle='--',
                alpha=0.7
            )

        # plot hit rate
        plt.title(
            "Instant Hit Rate Over Time",
            fontsize=18
        )
        plt.xlabel(
            "Request Index",
            fontsize=16
        )
        plt.ylabel(
            "Hit Rate (%)",
            fontsize=16
        )
        plt.legend()

        # second subplot: miss rate
        plt.subplot(2, 1, 2)
        for result in results:
            # extract results
            policy = result['policy']
            timeline = result['timeline']
            x = [
                point['index']
                for point in timeline
            ]
            y_miss = [
                100 - point['instant_hit_rate']
                for point in timeline
            ]
            plt.plot(
                x,
                y_miss,
                label=f"{policy} (miss)",
                linestyle='-',
                alpha=0.7
            )

        # plot miss rate
        plt.title(
            "Instant Miss Rate Over Time",
            fontsize=18
        )
        plt.xlabel(
            "Request Index",
            fontsize=16
        )
        plt.ylabel(
            "Miss Rate (%)",
            fontsize=16
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            "hit_miss_plot.png",
            format='png'
        )
        plt.show()
        plt.close()
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")


    # show a successful message
    info("🟢 Hit rate and miss rate plot built.")