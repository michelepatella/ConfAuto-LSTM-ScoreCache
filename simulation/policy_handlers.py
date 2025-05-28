import numpy as np
from torch.nn.functional import softmax
from utils.AccessLogsDataset import AccessLogsDataset
from utils.dataloader_utils import create_data_loader
from utils.inference_utils import _infer_batch, calculate_confidence_intervals
from utils.log_utils import info


def _find_key_candidates(
    all_outputs,
    upper_ci,
    lower_ci,
    config_settings
):
    """
    Method to find key candidates to be putted into the cache,
    based on how far predictions are w.r.t. the current timestamp.
    :param all_outputs: The outputs from the model.
    :param upper_ci: The upper confidence interval bound.
    :param lower_ci: The lower confidence interval bound.
    :param config_settings: The configuration settings.
    :return: The key candidates.
    """
    # build two matrices
    # probability matrix: [t, k] -> probability of using k at time t
    # confidence matrix: [t, k] -> confidence of the prediction k at time t
    num_steps = len(all_outputs)
    num_keys = len(all_outputs[0])
    prob_matrix = np.zeros((num_steps, num_keys))
    conf_matrix = np.zeros((num_steps, num_keys))

    # fill the matrices
    for t in range(num_steps):
        # take the probabilities at time step t
        probs = softmax(
            all_outputs[t],
            dim=0
        ).cpu().numpy()

        # calculate the confidence at time step t
        conf = (1 / (upper_ci[t] - lower_ci[t])).cpu().numpy()

        # fill both matrices
        prob_matrix[t] = probs
        conf_matrix[t] = conf

    scores = {}
    # for each key calculate a score to penalize
    # too temporally distant predictions
    for k in range(num_keys):
        score = 0.0
        for t in range(num_steps):
            decay = np.exp(-0.7 * t)
            score += prob_matrix[t, k] * conf_matrix[t, k] * decay
        scores[k] = score

    # select top-cache_size keys
    top_keys = sorted(
        scores,
        key=scores.get,
        reverse=True
    )[:config_settings.cache_size]

    return top_keys, prob_matrix, conf_matrix


def handle_lstm_policy(
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
    if (
        state['access_counter'] >= config_settings.prediction_interval and
        state['inference_start_idx'] < len(testing_set)
    ):
        # define a mobile window sliding over the testing set
        start_idx = state.get('inference_start_idx', 0)
        end_idx = start_idx + config_settings.prediction_interval
        testing_window_df = testing_set.data.iloc[start_idx:end_idx]
        testing_window_dataset = AccessLogsDataset.from_dataframe(
            testing_window_df,
            config_settings
        )

        # create a testing loader containing data
        # within the mobile window
        testing_loader = create_data_loader(
            testing_window_dataset,
            config_settings.testing_batch_size,
            shuffle=False
        )

        # make inference on this loader
        _, _, _, all_outputs, all_vars = _infer_batch(
            model,
            testing_loader,
            criterion,
            device,
            config_settings.mc_dropout_num_samples
        )

        # calculate CIs related to the predictions
        lower_ci, upper_ci = calculate_confidence_intervals(
            all_outputs,
            all_vars,
            config_settings
        )

        # identify candidate keys
        top_keys, prob_matrix, conf_matrix = _find_key_candidates(
            all_outputs,
            upper_ci,
            lower_ci,
            config_settings
        )

        # put the candidate keys into the cache
        for k in top_keys:
            prob_agg = np.sum(prob_matrix[:, k]) / len(all_outputs)
            conf_agg = np.mean(conf_matrix[:, k])
            cache.put(k, prob_agg, conf_agg, current_time)

        # update state variables
        state['access_counter'] = 0
        state['inference_start_idx'] = end_idx


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