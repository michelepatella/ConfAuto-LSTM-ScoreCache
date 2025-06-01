from cachetools import LRUCache, LFUCache, FIFOCache
from simulation.LSTMCache import LSTMCache
from simulation.RandomCache import RandomCache
from simulation.CacheWrapper import CacheWrapper
from simulation.simulator import simulate_cache_policy
from utils.graph_utils import plot_hit_miss_rate_over_time
from utils.log_utils import info
from utils.report_utils import print_system_evaluation_report


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
                config_settings
            ),
            'LFU': CacheWrapper(
                LFUCache,
                config_settings
            ),
            'FIFO': CacheWrapper(
                FIFOCache,
                config_settings
            ),
            'RANDOM': RandomCache(config_settings),
            'LSTM': LSTMCache(config_settings),
            'LSTM+CI': LSTMCache(config_settings)
        }

        # run simulations
        results = []
        for policy, cache in strategies.items():
            # simulate a cache policy and save results
            result = simulate_cache_policy(
                cache,
                policy,
                config_settings
            )
            results.append(result)

    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while running cache simulations: {e}.")

    # show results
    print_system_evaluation_report(results)

    # show hit rate and miss rate plot
    plot_hit_miss_rate_over_time(results)

    # print a successful message
    info("✅ Cache simulations completed.")