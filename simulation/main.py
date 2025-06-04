from simulation.caches.utils.CacheMetricsLogger import CacheMetricsLogger
from simulation.caches.lstm_cache.LSTMCache import LSTMCache
from simulation.caches.baseline_caches.RandomCache import RandomCache
from simulation.caches.utils.CacheWrapper import CacheWrapper
from simulation.caches.baseline_caches.FIFOCache import FIFOCache
from simulation.caches.baseline_caches.LFUCache import LFUCache
from simulation.caches.baseline_caches.LRUCache import LRUCache
from simulation.runner.simulation_runner import run_cache_simulation
from simulation.evaluation.visualization.simulation_plotter import plot_hit_miss_rate_over_time
from utils.logs.log_utils import info
from simulation.evaluation.visualization.simulation_reporter import generate_caches_evaluation_report


def run_simulations(config_settings):
    """
    Method to orchestrate simulation of cache strategies.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Cache simulations started...")

    try:
        # setup cache strategies
        strategies = {
            'LRU': CacheWrapper(
                LRUCache,
                CacheMetricsLogger(),
                config_settings,

            ),
            'LFU': CacheWrapper(
                LFUCache,
                CacheMetricsLogger(),
                config_settings
            ),
            'FIFO': CacheWrapper(
                FIFOCache,
                CacheMetricsLogger(),
                config_settings
            ),
            'RANDOM': RandomCache(
                None,
                CacheMetricsLogger(),
                config_settings
            ),
            'LSTM': LSTMCache(
                None,
                CacheMetricsLogger(),
                config_settings
            )
        }

        # run simulations
        results = []
        for policy, cache in strategies.items():
            # simulate a cache policy and save results
            result = run_cache_simulation(
                cache,
                policy,
                cache.metrics_logger,
                config_settings
            )
            results.append(result)

    except (
            KeyError,
            TypeError,
            ValueError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while running cache simulations: {e}.")

    # show results
    generate_caches_evaluation_report(results)

    # show hit rate and miss rate plot
    plot_hit_miss_rate_over_time(results)

    # print a successful message
    info("✅ Cache simulations completed.")