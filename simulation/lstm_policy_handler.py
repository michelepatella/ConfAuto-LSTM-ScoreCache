import random
import time
import numpy as np
from torch.nn.functional import softmax
from utils.simulation_utils import search_key
from utils.data.AccessLogsDataset import AccessLogsDataset
from utils.model.inference_utils import calculate_confidence_intervals, autoregressive_rollout
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
                # calculate the final score as a combination of
                # probability of a key of being used and CIs related
                # to that prediction
                score += prob_matrix[t, k] * (1 + (conf_matrix[t, k] - 0.5))
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
            IndexError,
            ZeroDivisionError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while calculating key scores: {e}.")

    # print a successful message
    info("🟢 Key scores calculated.")

    return scores


def _find_key_candidates(
        all_outputs,
        upper_ci,
        lower_ci
):
    """
    Method to find key candidates to be inserted into the cache.
    :param all_outputs: The outputs from the model.
    :param upper_ci: The upper confidence interval bound.
    :param lower_ci: The lower confidence interval bound.
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
        prob_matrix = np.zeros((
            num_steps,
            num_keys
        ))
        conf_matrix = np.zeros((
            num_steps,
            num_keys
        ))

        # fill the matrices
        for t in range(num_steps):
            # take the probabilities at time step t
            probs = softmax(
                all_outputs[t],
                dim=0
            ).cpu().numpy()
            prob_matrix[t] = probs

            # calculate the confidence at time step t
            conf = (
                    1 / (upper_ci[t] - lower_ci[t] + 1e-6)
            ).cpu().numpy()
            conf_matrix[t] = conf

        # normalize confidence matrix to [0,1]
        min_conf = np.min(conf_matrix)
        max_conf = np.max(conf_matrix)
        conf_range = max_conf - min_conf \
            if max_conf != min_conf \
            else 1.0
        conf_matrix = (conf_matrix - min_conf) / conf_range

        # calculate scores for the keys
        scores = _calculate_key_scores(
            num_keys,
            num_steps,
            prob_matrix,
            conf_matrix
        )

        # select the keys
        keys = list(scores.keys())

    except (
            IndexError,
            ZeroDivisionError,
            TypeError,
            KeyError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while finding key candidates: {e}.")

    # print a successful message
    info("🟢 Key candidates found.")

    return keys, scores


def _handle_cold_start(
        cache,
        current_time,
        config_settings
):
    """
    Method to handle cache cold start.
    :param cache: The cache for which to handle cold start.
    :param current_time: The current time.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Cold start handling started...")

    try:
        # select random keys
        all_possible_keys = list(range(
            config_settings.num_keys
        ))
        random_keys = random.sample(
            all_possible_keys,
            min(
                config_settings.cache_size,
                len(all_possible_keys)
            )
        )

        # fill the cache with these keys having 0 score
        for k in random_keys:
            cache.put(
                k,
                0.0,
                current_time,
                cold_start=True,
                config_settings=config_settings
            )
    except (
        NameError,
        AttributeError,
        TypeError,
        ValueError,
        KeyError
    ) as e:
        raise RuntimeError(f"❌ Error while handling cache cold start: {e}.")

    # print a successful message
    info("🟢 Cold start handled.")


def _extract_seed_seq(
        current_idx,
        testing_set,
        config_settings
):
    """
    Method to extract seed sequence used by
     autoregressive rollout.
    :param current_idx: The current request index.
    :param testing_set: The testing set.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Seed sequence extraction started...")

    try:
        # define a mobile window sliding over the testing set
        start_idx = current_idx - config_settings.seq_len + 1
        end_idx = current_idx + 1
        testing_window_df = testing_set.data.iloc[start_idx:end_idx + 1]

        # check if enough data contained in the sliding window
        if len(testing_window_df) < config_settings.seq_len:
            return

        # create a dataset from the sliding window
        testing_window_dataset = AccessLogsDataset.from_dataframe(
            testing_window_df,
            config_settings
        )

        # check if there is at least one element in the
        # dataset created
        if len(testing_window_dataset) == 0:
            return

        # extract seed from the testing window
        seed_seq = testing_window_dataset.__getitem__(
            len(testing_window_dataset) - 1
        )

    except (
        NameError,
        AttributeError,
        TypeError,
        IndexError
    ) as e:
        raise RuntimeError(f"❌ Error while extracting seed sequence: {e}.")

    # print a successful message
    info("🟢 Seed sequence extracted.")

    return seed_seq


def handle_lstm_cache_policy(
        cache,
        key,
        current_time,
        current_idx,
        counters,
        device,
        model,
        testing_set,
        config_settings
):
    """
    Method to handle the LSTM-based cache policy.
    :param cache: The cache.
    :param key: The current key.
    :param current_time: The current time.
    :param current_idx: The current index.
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
        start_time = None
        num_insertion = 0

        # search the key into the cache
        _ = search_key(
            cache,
            key,
            current_time,
            counters
        )

        # if it's not time to prefetch (no enough data)
        if current_idx < config_settings.seq_len:

            # handle cold start
            _handle_cold_start(
                cache,
                current_time,
                config_settings
            )

        elif (
            current_idx >= config_settings.seq_len and
            current_idx % config_settings.prediction_interval == 0
        ):
            # if it's the first time we do prefetch
            # save the number of hits of random policy
            # during cold start
            if current_idx == config_settings.seq_len:
                print(counters['hits'])
                # count the no. of hits during cold start
                counters['hits_cold_start'] = counters['hits']

                # remove all cached keys by the random policy
                for k in list(cache.store.keys()):
                    cache.evict_key(k)

            # keep track of autoregression and CIs calculation time
            start_time = time.perf_counter()

            # it's time to prefetch
            # extract seed sequence
            seed_seq = _extract_seed_seq(
                current_idx,
                testing_set,
                config_settings
            )

            # compute rollout
            (
                all_outputs,
                all_vars
            ) = autoregressive_rollout(
                model,
                seed_seq,
                device,
                config_settings
            )

            # calculate CIs related to the predictions
            (
                lower_ci,
                upper_ci
            ) = calculate_confidence_intervals(
                all_outputs,
                all_vars,
                config_settings
            )

            # identify keys and scores thereof
            (
                keys,
                scores
            ) = _find_key_candidates(
                all_outputs,
                upper_ci,
                lower_ci
            )

            # put the keys into the cache
            for k in keys:
                score = scores[k]
                is_cached = cache.put(
                    k,
                    score,
                    current_time,
                )

                # increase the number of prefetched keys
                if is_cached:
                    num_insertion += 1

    except (
            IndexError,
            KeyError,
            ValueError,
            AttributeError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while handling LSTM-based cache policy: {e}.")

    # print a successful message
    info(f"🟢 LSTM-based cache policy management completed.")

    if start_time is not None:
        return (
            time.perf_counter() - start_time,
            num_insertion
        )
    else:
        return (
            0,
            num_insertion
        )