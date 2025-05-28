from simulation.policy_handlers import handle_random_policy, handle_default_policy, handle_lstm_policy
from simulation.preprocessing import preprocess_data
from simulation.setup import simulation_setup


def simulate(cache, policy_name):
    """
    Method to orchestrate cache simulation.
    :param cache: The cache to simulate.
    :param policy_name: The cache policy name to use.
    :return: The hit rate and miss rate in terms of %.
    """
    # setup for simulation
    (
        CACHE_SIZE,
        TTL,
        df,
        hits,
        misses,
        time_map,
        access_counter,
        PREDICTION_INTERVAL,
        all_keys_seen
    ) = simulation_setup()

    # for each request
    for _, row in df.iterrows():

        # extrapolate key and timestamp
        key, current_time = preprocess_data(row)

        # if the LSTM cache is being used
        if policy_name == 'LSTM':
            handle_lstm_policy(
                cache,
                key,
                current_time,
                access_counter,
                PREDICTION_INTERVAL,
                hits,
                misses
            )

        # if the random cache is being used
        if policy_name == 'RANDOM':
            handle_random_policy(
                cache,
                key,
                current_time,
                TTL,
                hits,
                misses
            )
        else:
            # all the other caching policies (LRU, LFU, and FIFO)
            handle_default_policy(
                cache,
                key,
                current_time,
                TTL,
                time_map,
                hits,
                misses
            )

    # calculate hit rate and miss rate in terms of %
    total = hits + misses
    hit_rate = hits / total * 100
    miss_rate = misses / total * 100

    return {
        'policy': policy_name,
        'hit_rate': hit_rate,
        'miss_rate': miss_rate,
        'hits': hits,
        'misses': misses
    }