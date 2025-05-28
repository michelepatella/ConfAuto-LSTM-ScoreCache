from torch.nn.functional import softmax
from utils.inference_utils import _infer_batch, calculate_confidence_intervals
from utils.log_utils import info


def handle_lstm_policy(
        cache,
        key,
        current_time,
        state,
        counters,
        device,
        criterion,
        model,
        testing_loader,
        testing_set,
        config_settings
):
    """
    Method to handle the confident-aware LSTM-based cache.
    :param cache: The cache.
    :param key: The current key.
    :param current_time: The current time.
    :param state: The state variable.
    :param counters: The hits and misses counters.
    :param device: The device to be used while inferring.
    :param criterion: The loss function to be used while inferring.
    :param model: The model to infer.
    :param testing_loader: The testing loader.
    :param testing_set: The testing set.
    :param config_settings: The configuration settings.
    :return:
    """
    # increase the counter of LSTM cache usage
    state['access_counter'] += 1

    # check if the cache contains the key
    if cache.contains(key, current_time):
        # increment cache hits
        counters['hits'] += 1
        info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | HIT")
    else:
        # increment cache misses
        counters['misses'] += 1
        info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | MISS")

    # if it's time to infer
    if state['access_counter'] >= config_settings.prediction_interval:
        # make inference
        _, _, _, all_outputs, all_vars = _infer_batch(
            model,
            testing_loader,
            criterion,
            device,
            config_settings.mc_dropout_num_samples
        )

        # calculate CIs
        lower_ci, upper_ci = calculate_confidence_intervals(
            all_outputs, all_vars, config_settings
        )

        # update the cache for each key
        for idx, (output, low_ci, high_ci) in (
                enumerate(zip(all_outputs, lower_ci, upper_ci))):

            # extract the probability by applying softmax
            prob = (
                softmax(output, dim=0)
                    .cpu().numpy()
            )

            # extract the key id and the related probability
            key_id = testing_set.iloc[idx]['request']
            prob = prob[key_id]

            # update the cache
            cache.update_prediction_for_key(
                key_id,
                prob,
                (high_ci - low_ci).mean().item(),
                current_time
            )

        # reset the counter of LSTM cache usage
        state['access_counter'] = 0


def handle_random_policy(
        cache,
        key,
        current_time,
        counters,
        config_settings
):
    """
    Method to handle the Random cache.
    :param cache: The cache.
    :param key: The current key.
    :param current_time: The current time.
    :param counters: Hits and misses counters.
    :param config_settings: The configuration settings.
    :return:
    """
    # check if the key is in the cache
    if cache.contains(key, current_time):
        # increment cache hits
        counters['hits'] += 1
        info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | HIT")
    else:
        # increment cache misses
        counters['misses'] += 1
        info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | MISS")
    # put the key in the cache
    cache.put(
        key,
        config_settings.ttl_base,
        current_time
    )


def handle_default_policy(
        cache,
        key,
        current_time,
        counters,
        config_settings
):
    """
    Method to handle the LRU, LFU, and LIFO caches.
    :param cache: The cache.
    :param key: The current key.
    :param current_time: The current time.
    :param counters: Hits and misses counters.
    :param config_settings: The configuration settings.
    :return:
    """
    # check if the key is in the cache and
    # its TTL is not expired
    if cache.contains(key, current_time):
        # increment cache hits
        counters['hits'] += 1
        info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | HIT")
    else:
        # increment cache misses, set the key into the cache,
        # and assign it a TTL
        counters['misses'] += 1
        cache.put(key, config_settings.ttl_base, current_time)

        info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | MISS")