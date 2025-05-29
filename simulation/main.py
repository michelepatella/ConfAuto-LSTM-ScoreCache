from tqdm import tqdm
from simulation.lstm_policy_handler import handle_lstm_cache_policy
from simulation.traditional_policy_handler import handle_traditional_cache_policy
from simulation.preprocessing import preprocess_data
from utils.AccessLogsDataset import AccessLogsDataset
from utils.dataloader_utils import dataloader_setup
from utils.log_utils import info, debug
from utils.model_utils import trained_model_setup


def simulate(
        cache,
        policy_name,
        config_settings
):
    """
    Method to orchestrate cache simulation.
    :param cache: The cache object to simulate.
    :param policy_name: The cache policy name to use.
    :param config_settings: The configuration settings.
    :return: The hit rate and miss rate in terms of %.
    """
    # initial message
    info("🔄 Cache simulation started...")

    # debugging
    debug(f"⚙️Policy: {policy_name}.")

    # initialize data
    global device, criterion, model
    counters = {
        'hits': 0,
        'misses': 0
    }
    state = {
        'access_counter': 0,
        'inference_start_idx': 0
    }

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

        # if the LSTM cache is being used
        if policy_name == 'LSTM':
            handle_lstm_cache_policy(
                cache,
                key,
                current_time,
                state,
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
    info("✅ Cache simulation completed.")

    return {
        'policy': policy_name,
        'hit_rate': hit_rate,
        'miss_rate': miss_rate,
        'hits': counters['hits'],
        'misses': counters['misses']
    }