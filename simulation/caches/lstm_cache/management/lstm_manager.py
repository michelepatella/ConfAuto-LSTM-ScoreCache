import time
from simulation.caches.lstm_cache.management.cold_start_manager import manage_cold_start
from simulation.caches.lstm_cache.key_selection.key_candidates_finder import find_key_candidates
from simulation.caches.utils.seed_seq_extractor import extract_seed_seq
from simulation.caches.utils.key_finder import find_key
from simulation.prefetching.autoregression import autoregressive_rollout
from simulation.prefetching.confidence_interval_calculator import calculate_confidence_interval
from utils.logs.log_utils import info


def manage_lstm_cache(
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
        _ = find_key(
            cache,
            key,
            current_time,
            counters
        )

        # if it's not time to prefetch (no enough data)
        if current_idx < config_settings.seq_len:

            # handle cold start
            manage_cold_start(
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
                # count the no. of hits during cold start
                counters['hits_cold_start'] = counters['hits']

                # remove all cached keys by the random policy
                for k in list(cache.store.keys()):
                    cache.evict_key(k)

            # keep track of autoregression and CIs calculation time
            start_time = time.perf_counter()

            # it's time to prefetch
            # extract seed sequence
            seed_seq = extract_seed_seq(
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
            ) = calculate_confidence_interval(
                all_outputs,
                all_vars,
                config_settings
            )

            # identify keys and scores thereof
            (
                keys,
                scores
            ) = find_key_candidates(
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