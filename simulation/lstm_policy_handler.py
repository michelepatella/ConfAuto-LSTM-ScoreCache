import random
import numpy as np
from torch.nn.functional import softmax
from utils.AccessLogsDataset import AccessLogsDataset
from utils.inference_utils import calculate_confidence_intervals, autoregressive_rollout
from utils.log_utils import info


def _calculate_key_scores(
        num_keys,
        num_steps,
        prob_matrix,
        conf_matrix,
        confidence_aware
):
    """
    Method to calculate key scores.
    :param num_keys: The number of keys.
    :param num_steps: The number of steps.
    :param prob_matrix: The probability matrix.
    :param conf_matrix: The confidence matrix.
    :param confidence_aware: Specifies whether to use CIs.
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
                # calculate the final score as a combination of
                # probability of a key of being used and CIs related
                # to that prediction
                if confidence_aware:
                    score += prob_matrix[t, k] * np.log1p(conf_matrix[t, k])
                else:
                    score += prob_matrix[t, k]
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
        confidence_aware
):
    """
    Method to find key candidates to be inserted into the cache.
    :param all_outputs: The outputs from the model.
    :param upper_ci: The upper confidence interval bound.
    :param lower_ci: The lower confidence interval bound.
    :param confidence_aware: Specifies whether to use CIs.
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
            prob_matrix[t] = probs

            if confidence_aware:
                # calculate the confidence at time step t
                conf = (1 / (upper_ci[t] - lower_ci[t] + 1e-6)).cpu().numpy()
                conf_matrix[t] = conf

        # calculate scores for the keys
        scores = _calculate_key_scores(
            num_keys,
            num_steps,
            prob_matrix,
            conf_matrix,
            confidence_aware
        )

        # select the keys
        keys = list(scores.keys())

    except (
            IndexError, ZeroDivisionError, TypeError,
            KeyError, AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while finding key candidates: {e}.")

    # print a successful message
    info("🟢 Key candidates found.")

    return keys, scores


def handle_lstm_cache_policy(
        cache,
        key,
        current_time,
        current_idx,
        counters,
        device,
        model,
        testing_set,
        config_settings,
        confidence_aware
):
    """
    Method to handle the confident-aware LSTM-based cache policy.
    :param cache: The cache.
    :param key: The current key.
    :param current_time: The current time.
    :param current_idx: The current index.
    :param counters: The hits and misses counters.
    :param device: The device to be used while inferring.
    :param model: The model to use to infer.
    :param testing_set: The testing set.
    :param config_settings: The configuration settings.
    :param confidence_aware: Specifies whether to use CIs.
    :return:
    """
    # initial message
    global upper_ci, lower_ci
    info(f"🔄 LSTM-based cache policy management started...")

    try:

        # check if the cache contains the key
        if cache.contains(key, current_time):
            # increment cache hits
            counters['hits'] += 1
            info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | HIT")
        else:
            # increment cache misses
            counters['misses'] += 1
            info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | MISS")

        # if it's not time to prefetch (no enough data)
        if current_idx < config_settings.seq_len:
            # fill the cache randomly
            all_possible_keys = list(range(config_settings.num_keys))
            random_keys = random.sample(
                all_possible_keys,
                min(config_settings.cache_size, len(all_possible_keys))
            )
            # assign a score 0.0 to these keys
            for k in random_keys:
                cache.put(
                    k,
                    0.0,
                    current_time,
                    cold_start=True,
                    config_settings=config_settings
                )

        elif (
            current_idx >= config_settings.seq_len and
            current_idx % config_settings.prediction_interval == 0
        ):

            # define a mobile window sliding over the testing set
            start_idx = current_idx - config_settings.seq_len + 1
            end_idx = current_idx + 1
            testing_window_df = testing_set.data.iloc[start_idx:end_idx+1]

            if len(testing_window_df) < config_settings.seq_len:
                return

            # create a dataset from this window
            testing_window_dataset = AccessLogsDataset.from_dataframe(
                testing_window_df,
                config_settings
            )

            # check if there is at least one element
            if len(testing_window_dataset) == 0:
                return

            # extract seed from the testing window
            seed_seq = testing_window_dataset.__getitem__(
                len(testing_window_dataset) - 1
            )

            # compute rollout
            all_outputs, all_vars = autoregressive_rollout(
                model,
                seed_seq,
                device,
                config_settings,
                confidence_aware
            )

            if confidence_aware:
                # calculate CIs related to the predictions
                lower_ci, upper_ci = calculate_confidence_intervals(
                    all_outputs,
                    all_vars,
                    config_settings
                )

            # identify keys and scores thereof
            keys, scores = _find_key_candidates(
                all_outputs,
                upper_ci,
                lower_ci,
                confidence_aware
            )

            # put the keys into the cache
            for k in keys:
                score = scores[k]
                cache.put(
                    k,
                    score,
                    current_time
                )

    except (IndexError, KeyError, ValueError, AttributeError, TypeError) as e:
        raise RuntimeError(f"❌ Error while handling LSTM-based cache policy: {e}.")

    # print a successful message
    info(f"🟢 LSTM-based cache policy management completed.")