from simulation.evaluation.metrics_calculator import calculate_eviction_mistake_rate, calculate_prefetch_hit_rate, \
    calculate_hit_miss_rate, calculate_prefetching_avg_latency
from utils.logs.log_utils import info


def evaluate_cache(
        counters,
        tot_prefetch,
        autoregressive_latencies,
        metrics_logger
):
    """
    Method to orchestrate cache metrics calculation.
    :param counters: A counter used while simulating a cache policy.
    :param tot_prefetch: The total number of prefetches.
    :param autoregressive_latencies: The autoregressive latencies.
    :param metrics_logger: The metrics logger.
    :return: All the computed cache metrics.
    """
    # initial message
    info("🔄 Cache metrics calculation started...")

    # calculate hit rate and miss rate
    (
        hit_rate,
        miss_rate
    ) = calculate_hit_miss_rate(
        counters
    )

    # component evaluation
    prefetch_hit_rate = calculate_prefetch_hit_rate(
        counters['hits'] - counters['hits_cold_start'],
        tot_prefetch
    )
    eviction_mistake_rate = calculate_eviction_mistake_rate(
        metrics_logger
    )

    # calculate avg prefetching average
    avg_prefetching_latency = calculate_prefetching_avg_latency(
        autoregressive_latencies
    )

    # show a successful message
    info("🟢 Cache metrics calculated.")

    return (
        hit_rate,
        miss_rate,
        prefetch_hit_rate,
        eviction_mistake_rate,
        avg_prefetching_latency
    )