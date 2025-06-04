from utils.logs.log_utils import info


def generate_caches_evaluation_report(results):
    """
    Method to show system evaluation report.
    :param results: The system evaluation results.
    :return:
    """
    # initial message
    info("🔄 System simulation report printing started...")

    try:
        # title
        print("\n" + "=" * 155)
        print(" " * 30 + "Overall System Evaluation Report")
        print("=" * 155 + "\n")

        # header with additional metrics
        header = (
            f"{'Policy':<25} | {'Hit Rate (%)':>12} | {'Miss Rate (%)':>13} | "
            f"{'Avg Prefetching Latency (s)':>27} | {'Eviction Mistake Rate (%)':>26} | "
            f"{'Prefetch Hit Rate (%)':>22}"
        )
        print(header)
        print("-" * len(header))

        # results including all metrics
        for res in results:
            eviction_rate = (
                f"{res['eviction_mistake_rate'] * 100:.2f}"
                if res.get('eviction_mistake_rate') is not None else "N/A"
            )
            prefetch_rate = (
                f"{res['prefetch_hit_rate'] * 100:.2f}"
                if res.get('prefetch_hit_rate') is not None else "N/A"
            )

            print(
                f"{res['policy']:<25} | "
                f"{res['hit_rate']:12.2f} | "
                f"{res['miss_rate']:13.2f} | "
                f"{res['avg_prefetching_latency']:27.6f} | "
                f"{eviction_rate:26} | "
                f"{prefetch_rate:22}"
            )
        print("\n" + "=" * 155 + "\n")

    except (
            TypeError,
            KeyError,
            ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while printing system simulation report: {e}.")

    # print a successful message
    info("🟢 System simulation report printed.")