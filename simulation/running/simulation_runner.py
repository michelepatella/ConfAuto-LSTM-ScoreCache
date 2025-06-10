from tqdm import tqdm
from simulation.caches.lstm_cache.management.lstm_manager import manage_lstm_cache
from simulation.running.simulation_setup import simulation_setup
from simulation.running.simulation_tracer import trace_hits_misses
from simulation.utils.row_preprocessor import preprocess_row
from simulation.caches.utils.key_finder import find_key
from utils.logs.log_utils import info, debug
from simulation.evaluation.cache_evaluator import evaluate_cache


def run_cache_simulation(
        cache,
        policy_name,
        metrics_logger,
        config_settings
):
    """
    Method to simulate a cache policy.
    :param cache: The cache object to simulate.
    :param policy_name: The cache policy name to use.
    :param metrics_logger: The metrics logger.
    :param config_settings: The configuration settings.
    :return: The hit rate and miss rate in terms of %.
    """
    # initial message
    info(f"🔄 {policy_name} policy simulation started...")

    # debugging
    debug(f"⚙️ Policy: {policy_name}.")

    # setup for simulation
    (
        counters,
        timeline,
        recent_hits,
        autoregressive_latencies,
        window,
        testing_set,
        testing_loader,
        device,
        criterion,
        model
    ) = simulation_setup(
        policy_name,
        config_settings
    )

    tot_prefetch = 0
    # for each request
    for idx in tqdm(
            range(len(testing_set)),
            desc=f"Simulating {policy_name}"
    ):
        try:
            # extract the row from the dataset
            row = testing_set[idx]

            # keep track of the no. of hits so far
            prev_hits = counters['hits']
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except IndexError as e:
            raise IndexError(f"IndexError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except KeyError as e:
            raise KeyError(f"KeyError: {e}.")
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except ValueError as e:
            raise ValueError(f"ValueError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # extrapolate timestamp and key from the row
        (
            current_time,
            key
        ) = preprocess_row(row)

        # debugging
        debug(f"⚙️ Current time: {current_time} - Key: {key}.")

        # if the LSTM cache is being used
        if policy_name == 'LSTM':
            (
                autoregressive_latency,
                num_prefetch
            )= manage_lstm_cache(
                cache,
                key,
                current_time,
                idx,
                counters,
                device,
                model,
                testing_set,
                config_settings
            )

            # update no. of prefetch
            tot_prefetch += num_prefetch

            # store cache latency
            autoregressive_latencies.append(
                autoregressive_latency
            )

        # if the traditional cache (LRU, LFU, FIFO, or RANDOM) is being used
        else:
            # search the key into the cache
            is_hit = find_key(
                cache,
                key,
                current_time,
                counters
            )

            if not is_hit:
                # put a key in cache
                cache.put(
                    key,
                    current_time
                )

        # update number of hits and misses
        (
            recent_hits,
            timeline
        ) = trace_hits_misses(
            counters,
            prev_hits,
            recent_hits,
            window,
            idx,
            timeline
        )

    # compute cache metrics
    (
        hit_rate,
        miss_rate,
        prefetch_hit_rate,
        eviction_mistake_rate,
        avg_prefetching_latency
    ) = evaluate_cache(
        counters,
        tot_prefetch,
        autoregressive_latencies,
        metrics_logger
    )

    # print a successful message
    info(f"🟢 {policy_name} policy simulation completed.")

    return {
        'policy': policy_name,
        'hit_rate': hit_rate,
        'miss_rate': miss_rate,
        'hits': counters['hits'],
        'misses': counters['misses'],
        'avg_prefetching_latency': avg_prefetching_latency,
        'timeline': timeline,
        'prefetch_hit_rate': prefetch_hit_rate,
        'eviction_mistake_rate': eviction_mistake_rate
    }