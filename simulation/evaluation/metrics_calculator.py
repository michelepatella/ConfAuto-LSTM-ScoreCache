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


def calculate_prefetch_hit_rate(
        prefetch_hits,
        tot_prefetch,
):
    """
    Method to compute prefetch hit rate.
    :param prefetch_hits: The number of prefetch hits.
    :param tot_prefetch: Total number of prefetches.
    :return: The prefetch hit rate.
    """
    # initial message
    info("🔄 Prefetch hit rate calculation started...")

    try:
        if tot_prefetch > 0:
            # calculate prefetch hit rate
            prefetch_hit_rate =  prefetch_hits / tot_prefetch
        else:
            prefetch_hit_rate = 0.0

    except (
        AttributeError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        NameError
    ) as e:
        raise RuntimeError(f"❌ Error while computing prefetch hit rate: {e}.")

    # show a successful message
    info("🟢 Prefetch hit rate computed.")

    return prefetch_hit_rate


def calculate_hit_miss_rate(counters):
    """
    Method to calculate hit and miss rate.
    :param counters: A counter used while simulating a cache policy
    :return: The hit and miss rate in terms of %.
    """
    # initial message
    info("🔄 Hit and miss rate calculation started...")

    try:
        # calculate hit rate and miss rate in terms of %
        total = counters['hits'] + counters['misses']
        hit_rate = counters['hits'] / total * 100
        miss_rate = counters['misses'] / total * 100
    except (
            KeyError,
            ZeroDivisionError,
            TypeError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while calculating hit and miss rate: {e}.")

    # show a successful message
    info("🟢 Hit and miss rate calculated.")

    return hit_rate, miss_rate


def calculate_prefetching_avg_latency(
        autoregressive_latencies
):
    """
    Method to calculate prefetching average latency.
    :return: Prefetching average latency.
    """
    # initial message
    info("🔄 Prefetching average latency calculation started...")

    try:
        avg_prefetching_latency = 0
        if (
            autoregressive_latencies is not None and
            not len(autoregressive_latencies) == 0
        ):
            # filter only not None elements
            valid_latencies = [
                lat for lat in autoregressive_latencies
                if lat is not None
            ]
            if valid_latencies:
                # calculate average prefetching latency
                avg_prefetching_latency = (
                        sum(valid_latencies)/
                        len(valid_latencies)
                )
    except (
        TypeError,
        ZeroDivisionError,
        AttributeError,
        NameError,
        ValueError,
    ) as e:
        raise RuntimeError(f"❌ Error while calculating average prefetching latency: {e}.")

    # show a successful message
    info("🟢 Prefetching average latency calculated.")

    return avg_prefetching_latency