from utils.logs.log_utils import info


def calculate_eviction_mistake_rate(
        metrics_logger,
        mistake_window=300
):
    """
    Method to compute eviction mistake rate.
    :param metrics_logger: The metrics logger.
    :param mistake_window: The mistake window to consider.
    :return: The eviction mistake rate.
    """
    # initial message
    info("🔄 Eviction mistake rate calculation started...")

    try:
        # initialize data
        mistakes = 0
        total_eviction_events = 0

        # count mistakes within a temporal window
        for key, eviction_times in metrics_logger.evicted_keys.items():
            for eviction_time in eviction_times:
                total_eviction_events += 1
                # look for any future access after the current eviction
                future_accesses = [
                    t for t in metrics_logger.access_events.get(key, [])
                    if t > eviction_time and (t - eviction_time) <= mistake_window
                ]
                if future_accesses:
                    mistakes += 1
    except (
        AttributeError,
        TypeError,
        ZeroDivisionError
    ) as e:
        raise RuntimeError(f"❌ Error while computing eviction mistake rate: {e}.")

    # show a successful message
    info("🟢 Eviction mistake rate computed.")

    return mistakes / total_eviction_events \
        if total_eviction_events > 0 \
        else 0