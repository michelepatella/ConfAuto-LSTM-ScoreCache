import numpy as np
from torch.nn.functional import softmax
from utils.AccessLogsDataset import AccessLogsDataset
from utils.inference_utils import calculate_confidence_intervals, autoregressive_rollout
from utils.log_utils import info


def _calculate_key_scores(
        num_keys,
        num_steps,
        prob_matrix,
        conf_matrix
):
    """
    Method to calculate key scores.
    :param num_keys: The number of keys.
    :param num_steps: The number of steps.
    :param prob_matrix: The probability matrix.
    :param conf_matrix: The confidence matrix.
    :return: The key scores.
    """
    # initial message
    info("🔄 Key scores calculation started...")

    try:
        scores = {}
        # for each key calculate a score
        for k in range(num_keys):
            score = 0.0
            for t in range(num_steps):
                # the decay penalizes too distant temporal events
                decay = 1 / (1 + t)
                # calculate the final score as a combination of
                # probability of a key of being used, CIs related
                # to that prediction, and how distant the event is
                # w.r.t. the current time
                score += prob_matrix[t, k] * conf_matrix[t, k] * decay
            scores[k] = score

        # normalize scores in [0,1]
        min_score = min(scores.values())
        max_score = max(scores.values())
        score_range = max_score - min_score \
            if max_score != min_score \
            else 1.0
        scores = {
            k: (v - min_score) / score_range
            for k, v in scores.items()
        }

    except (
            IndexError, ZeroDivisionError, ValueError,
            TypeError, KeyError, AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while calculating key scores: {e}.")

    # print a successful message
    info("🟢 Key scores calculated.")

    return scores


def _find_key_candidates(
    all_outputs,
    upper_ci,
    lower_ci,
    config_settings
):
    """
    Method to find key candidates to be inserted into the cache.
    :param all_outputs: The outputs from the model.
    :param upper_ci: The upper confidence interval bound.
    :param lower_ci: The lower confidence interval bound.
    :param config_settings: The configuration settings.
    :return: The key candidates.
    """
    # initial message
    info("🔄 Search for key candidate started...")

    try:
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

        # calculate scores for the keys
        scores = _calculate_key_scores(
            num_keys,
            num_steps,
            prob_matrix,
            conf_matrix
        )

        # select top-cache_size keys
        top_keys = sorted(
            scores,
            key=scores.get,
            reverse=True
        )[:config_settings.cache_size]

    except (
            IndexError, ZeroDivisionError, TypeError,
            KeyError, AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while finding key candidates: {e}.")

    # print a successful message
    info("🟢 Key candidates found.")

    return top_keys, scores


def handle_lstm_cache_policy(
        cache,
        key,
        current_time,
        state,
        counters,
        device,
        model,
        testing_set,
        config_settings
):
    """
    Method to handle the confident-aware LSTM-based cache policy.
    :param cache: The cache.
    :param key: The current key.
    :param current_time: The current time.
    :param state: The state variable.
    :param counters: The hits and misses counters.
    :param device: The device to be used while inferring.
    :param model: The model to use to infer.
    :param testing_set: The testing set.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info(f"🔄 LSTM-based cache policy management started...")

    try:

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

        # if it's time to infer and we can do it
        if (
            state['access_counter'] >= config_settings.prediction_interval and
            state['inference_start_idx'] < len(testing_set)
        ):
            # define a mobile window sliding over the testing set
            start_idx = state.get('inference_start_idx', 0)
            end_idx = start_idx + config_settings.prediction_interval
            testing_window_df = testing_set.data.iloc[start_idx:end_idx]

            # create a dataset from this window
            testing_window_dataset = AccessLogsDataset.from_dataframe(
                testing_window_df,
                config_settings
            )

            # extract seed from the testing window
            seed_seq = testing_window_dataset.__getitem__(0)

            # compute rollout
            all_outputs, all_vars = autoregressive_rollout(
                model,
                seed_seq,
                config_settings,
                30,
                device
            )

            # calculate CIs related to the predictions
            lower_ci, upper_ci = calculate_confidence_intervals(
                all_outputs,
                all_vars,
                config_settings
            )

            # identify candidate keys to be inserted
            # into the cache
            top_keys, scores = _find_key_candidates(
                all_outputs,
                upper_ci,
                lower_ci,
                config_settings
            )

            # put the candidate keys into the cache
            for k in top_keys:
                score = scores[k]
                cache.put(k, score, current_time)

            # update state variables
            state['access_counter'] = 0
            state['inference_start_idx'] = end_idx

    except (IndexError, KeyError, ValueError, AttributeError, TypeError) as e:
        raise RuntimeError(f"❌ Error while handling LSTM-based cache policy: {e}.")

    # print a successful message
    info(f"🟢 LSTM-based cache policy management completed.")