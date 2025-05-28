from simulation.policy_handlers import handle_random_policy, handle_default_policy, handle_lstm_policy
from simulation.preprocessing import preprocess_data
from utils.AccessLogsDataset import AccessLogsDataset
from utils.dataloader_utils import dataloader_setup
from utils.model_utils import trained_model_setup


def simulate(
        cache,
        policy_name,
        config_settings
):
    """
    Method to orchestrate cache simulation.
    :param cache: The cache to simulate.
    :param policy_name: The cache policy name to use.
    :param config_settings: The configuration settings.
    :return: The hit rate and miss rate in terms of %.
    """
    # initialize data
    global device, criterion, model
    counters = {'hits': 0, 'misses': 0}
    state = {'access_counter': 0, 'inference_start_idx': 0}

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

        model.eval()
        model.to(device)

    # for each request
    for idx in range(len(testing_set)):

        # extract the row
        row = testing_set[idx]

        # extrapolate timestamp and key
        current_time, key = preprocess_data(row)

        # if the LSTM cache is being used
        if policy_name == 'LSTM':
            handle_lstm_policy(
                cache,
                key,
                current_time,
                state,
                counters,
                device,
                criterion,
                model,
                testing_set,
                config_settings
            )
        # if the random cache is being used
        elif policy_name == 'RANDOM':
            handle_random_policy(
                cache,
                key,
                current_time,
                counters,
                config_settings
            )
        else:
            # all the other caching policies (LRU, LFU, and FIFO)
            handle_default_policy(
                cache,
                key,
                current_time,
                counters,
                config_settings
            )

    # calculate hit rate and miss rate in terms of %
    total = counters['hits'] + counters['misses']
    hit_rate = counters['hits'] / total * 100
    miss_rate = counters['misses'] / total * 100

    return {
        'policy': policy_name,
        'hit_rate': hit_rate,
        'miss_rate': miss_rate,
        'hits': counters['hits'],
        'misses': counters['misses']
    }