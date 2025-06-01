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
        # define cache sizes and TTL values
        cache_size = {
            'small': 5,
            'medium': 10,
            'large': 15
        }
        ttl = {
            'small': 30,
            'medium': 60,
            'large': 120
        }

        # run simulations
        results = []
        for (size_key, size_val), (_, ttl_val) in zip(cache_size.items(), ttl.items()):
            # change config
            config_settings.cache_size = size_val
            config_settings.fixed_ttl = ttl_val
            config_settings.ttl_base = ttl_val

            # define strategies
            strategies = {
                'LRU': CacheWrapper(LRUCache, config_settings),
                'LFU': CacheWrapper(LFUCache, config_settings),
                'FIFO': CacheWrapper(FIFOCache, config_settings),
                'RANDOM': RandomCache(config_settings),
                'LSTM': LSTMCache(config_settings)
            }

            # simulate a cache policy and save results
            for policy, cache in strategies.items():
                # simulate and store result
                result = simulate_cache_policy(cache, policy, config_settings)
                result['cache_size'] = size_val
                result['ttl'] = ttl_val
                results.append(result)

    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while running cache simulations: {e}.")

    # show results
    print_system_evaluation_report(results)

    # show hit rate and miss rate plot
    plot_hit_miss_rate_over_time(results)

    # print a successful message
    info("✅ Cache simulations completed.")