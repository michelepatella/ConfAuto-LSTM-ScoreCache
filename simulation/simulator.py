from tqdm import tqdm
from simulation.lstm_policy_handler import handle_lstm_cache_policy
from simulation.preprocessing import preprocess_data
from utils.simulation_utils import search_key
from utils.AccessLogsDataset import AccessLogsDataset
from utils.dataloader_utils import dataloader_setup
from utils.log_utils import info, debug
from utils.metrics_utils import compute_eviction_mistake_rate, compute_ttl_success_rate, compute_prefetch_hit_rate, \
    calculate_hit_miss_rate, calculate_prefetching_avg_latency
from utils.model_utils import trained_model_setup


def _setup_simulation(
        policy_name,
        config_settings
):
    """
    Method to set up the simulation environment.
    :param policy_name: The policy name.
    :param config_settings: The configuration settings.
    :return: All the data needed to run the simulation.
    """
    # initial message
    info("🔄 Simulation setup started...")

    # initialize data
    (
        device,
        criterion,
        model
    ) = None, None, None
    counters = {
        'hits': 0,
        'misses': 0,
        'hits_cold_start': 0
    }
    timeline = []
    recent_hits = []
    prefetching_latency = []
    window = config_settings.prediction_interval

    # get the testing set
    (
        testing_set,
        testing_loader
    ) = dataloader_setup(
        "testing",
        config_settings.testing_batch_size,
        False,
        config_settings,
        AccessLogsDataset
    )

    # initial model setup, in case of LSTM cache
    if policy_name == 'LSTM':
        # setup for lstm cache
        (
            device,
            criterion,
            model
        ) = trained_model_setup(
            testing_loader,
            config_settings
        )

        try:
            model.eval()
            model.to(device)
        except (
                AttributeError,
                NameError,
                TypeError
        ) as e:
            raise RuntimeError(f"❌ Error while setting model evaluation and moving it to device: {e}.")

    # print a successful message
    info("🟢 Simulation setup completed.")

    return (
        counters,
        timeline,
        recent_hits,
        prefetching_latency,
        window,
        testing_set,
        testing_loader,
        device,
        criterion,
        model,
    )


def _trace_hits_misses(
        counters,
        prev_hits,
        recent_hits,
        window,
        idx,
        timeline
):
    """
    Method to trace hits and misses over time.
    :param counters: A counter used while simulating the cache policy.
    :param prev_hits: The previous number of hits.
    :param recent_hits: The recent number of hits.
    :param window: A window in which to calculate hits and misses.
    :param idx: The index of the current request.
    :param timeline: The timeline of hits and misses.
    :return: The number of recent hits and timeline updated.
    """
    # initial message
    info("🔄 Number of hits and misses counting started...")

    try:
        # keep track of the number of hits each time
        was_hit = counters['hits'] > prev_hits
        recent_hits.append(1 if was_hit else 0)
        if len(recent_hits) > window:
            recent_hits.pop(0)

        # calculate instant hit rate and overall average hit rate
        instant_hit_rate = counters['hits'] / (idx + 1)

        # store timeline data
        timeline.append({
            'index': idx,
            'instant_hit_rate': instant_hit_rate,
            'total_hits': counters['hits'],
            'total_misses': counters['misses']
        })
    except (
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
            ZeroDivisionError
    ) as e:
        raise RuntimeError(f"❌ Error while keeping track of the no. of hits and misses: {e}.")

    # print a successful message
    info("🟢 Number of hits and misses counted.")

    return recent_hits, timeline


def simulate_cache_policy(
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
        model,
    ) = _setup_simulation(
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
        except (
            NameError,
            IndexError,
            TypeError,
            KeyError,
            AttributeError,
            ValueError
        ) as e:
            raise RuntimeError(f"❌ Error while running simulation: {e}.")

        # extrapolate timestamp and key from the row
        (
            current_time,
            key
        ) = preprocess_data(row)

        # debugging
        debug(f"⚙️Current time: {current_time} - Key: {key}.")

        # if the LSTM cache is being used
        if policy_name == 'LSTM':
            (
                autoregressive_latency,
                num_prefetch
            )= handle_lstm_cache_policy(
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
            search_key(
                cache,
                key,
                current_time,
                counters
            )

            # put a key in cache
            cache.put(
                key,
                config_settings.fixed_ttl,
                current_time
            )

        # update number of hits and misses
        recent_hits, timeline = _trace_hits_misses(
            counters,
            prev_hits,
            recent_hits,
            window,
            idx,
            timeline
        )

    # calculate hit rate and miss rate
    (
        hit_rate,
        miss_rate
    ) = calculate_hit_miss_rate(
        counters
    )

    # component evaluation
    prefetch_hit_rate = compute_prefetch_hit_rate(
        counters['hits']-counters['hits_cold_start'],
        tot_prefetch
    )
    ttl_success_rate = compute_ttl_success_rate(
        metrics_logger
    )
    eviction_mistake_rate = compute_eviction_mistake_rate(
        metrics_logger
    )

    # calculate avg prefetching average
    avg_prefetching_latency = calculate_prefetching_avg_latency(
        autoregressive_latencies
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
        'ttl_success_rate': ttl_success_rate,
        'eviction_mistake_rate': eviction_mistake_rate
    }