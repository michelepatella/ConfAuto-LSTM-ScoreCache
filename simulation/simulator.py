from tqdm import tqdm
import time
from simulation.lstm_policy_handler import handle_lstm_cache_policy
from simulation.traditional_policy_handler import handle_traditional_cache_policy
from simulation.preprocessing import preprocess_data
from utils.AccessLogsDataset import AccessLogsDataset
from utils.dataloader_utils import dataloader_setup
from utils.log_utils import info, debug
from utils.model_utils import trained_model_setup


def simulate_cache_policy(
        cache,
        policy_name,
        config_settings
):
    """
    Method to simulate a cache policy.
    :param cache: The cache object to simulate.
    :param policy_name: The cache policy name to use.
    :param config_settings: The configuration settings.
    :return: The hit rate and miss rate in terms of %.
    """
    # initial message
    info(f"🔄 {policy_name} policy simulation started...")

    # debugging
    debug(f"⚙️Policy: {policy_name}.")

    # initialize data
    global device, criterion, model
    counters = {
        'hits': 0,
        'misses': 0
    }
    timeline = []
    recent_hits = []
    latencies = []
    window = config_settings.prediction_interval

    # get the testing set
    testing_set, testing_loader = dataloader_setup(
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
        ) = trained_model_setup(testing_loader, config_settings)

        try:
            model.eval()
            model.to(device)
        except (AttributeError, NameError, TypeError) as e:
            raise RuntimeError(f"❌ Error while setting model evaluation "
                               f"or moving it to device: {e}.")

    # for each request
    for idx in tqdm(
            range(len(testing_set)),
            desc=f"Simulating {policy_name}"
    ):
        # keep track of the start time
        start_time = time.perf_counter()

        try:
            # extract the row from the dataset
            row = testing_set[idx]
        except (IndexError, KeyError, TypeError, NameError) as e:
            raise RuntimeError(f"❌ Error while extracting the row"
                               f" from the dataset: {e}.")

        # extrapolate timestamp and key from the row
        current_time, key = preprocess_data(row)

        # debugging
        debug(f"⚙️Current time: {current_time} - Key: {key}.")

        # keep track of the no. of hits so far
        prev_hits = counters['hits']

        # if the LSTM cache is being used
        if policy_name == 'LSTM':
            handle_lstm_cache_policy(
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
        # if the traditional cache (LRU, LFU, FIFO, or RANDOM) is being used
        else:
            handle_traditional_cache_policy(
                cache,
                policy_name,
                key,
                current_time,
                counters,
                config_settings
            )

        # at the end, calculate the latency
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)

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
        except (KeyError, TypeError, AttributeError, IndexError, ZeroDivisionError) as e:
            raise RuntimeError(f"❌ Error while manipulating simulation data: {e}.")

    try:
        # calculate hit rate and miss rate in terms of %
        total = counters['hits'] + counters['misses']
        hit_rate = counters['hits'] / total * 100
        miss_rate = counters['misses'] / total * 100
    except (KeyError, ZeroDivisionError, TypeError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while calculating hit and miss rate: {e}.")

    # show results
    info(f"🎯 Hit Rate ({policy_name}): {hit_rate:.2f}%)")
    info(f"🚫 Miss Rate ({policy_name}): {miss_rate:.2f}%)")

    # print a successful message
    info(f"🟢 {policy_name} policy simulation completed.")

    return {
        'policy': policy_name,
        'hit_rate': hit_rate,
        'miss_rate': miss_rate,
        'hits': counters['hits'],
        'misses': counters['misses'],
        'avg_latency': sum(latencies)/len(latencies),
        'timeline': timeline
    }