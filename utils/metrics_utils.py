import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from utils.log_utils import info


def _calculate_top_k_accuracy(
        targets,
        outputs,
        config_settings
):
    """
    To calculate the top-k accuracy of the predictions.
    :param targets: The targets.
    :param outputs: The outputs of the model.
    :param config_settings: The configuration settings.
    :return: The top k-accuracy of the predictions.
    """
    # initial message
    info("🔄 Top-k accuracy computation started...")

    try:
        # prepare data
        outputs_tensor = torch.stack(outputs)

        # extract top-k predictions
        top_k_preds = (torch.topk(
            outputs_tensor,
            k=config_settings.top_k,
            dim=1
        ).indices.cpu().numpy())

        # initialize the no. of correct predictions
        correct = 0

        # count the correct predictions
        for i in range(len(targets)):

            # get the top-k predictions
            top_k_i = top_k_preds[i][:config_settings.top_k]

            # check if the target is contained into the
            # top-k predictions
            if targets[i] in top_k_i:
                correct += 1

        # calculate the accuracy
        accuracy = correct / len(targets)

    except (
            RuntimeError,
            IndexError,
            TypeError,
            ZeroDivisionError,
            ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while computing top-k accuracy: {e}.")

    # show a successful message
    info("🟢 Top-k accuracy computed.")

    return accuracy


def _calculate_kappa_statistic(
        targets,
        predictions
):
    """
    Method to calculate the kappa statistic.
    :param targets: The targets.
    :param predictions: The predictions coming from the model.
    :return: The kappa statistic.
    """
    # initial message
    info("🔄 Kappa statistic calculation started...")

    try:
        # calculate kappa statistic
        kappa = cohen_kappa_score(
            targets,
            predictions
        )
    except (
            ValueError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while calculating kappa statistic: {e}.")

    # show a successful message
    info("🟢 Kappa statistic calculated.")

    return kappa


def _compute_model_standalone_metrics(
        targets,
        predictions,
        outputs,
        config_settings
):
    """
    Method to compute metrics based on predictions and targets.
    :param targets: The targets.
    :param predictions: Predictions from model.
    :param outputs: Probabilities from model.
    :param config_settings: The configuration settings.
    :return: The computed metrics.
    """
    # initial message
    info("🔄 Metrics computation started...")

    try:
        # class-wise metrics
        class_report = classification_report(
            targets,
            predictions,
            output_dict=True,
            zero_division=0
        )

        # calculate the top-k accuracy
        top_k_accuracy = _calculate_top_k_accuracy(
            targets,
            outputs,
            config_settings
        )

        # compute the confusion matrix
        conf_matrix = confusion_matrix(
            targets,
            predictions
        )

        # calculate kappa statistic
        kappa_statistic = _calculate_kappa_statistic(
            targets,
            predictions
        )

    except (
            ValueError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while computing metrics: {e}.")

    # collect metrics
    metrics = {
        "class_report": class_report,
        "top_k_accuracy": top_k_accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "kappa_statistic": kappa_statistic
    }

    # show a successful message
    info("🟢 Metrics computed.")

    return metrics


def compute_eviction_mistake_rate(
        metrics_logger,
        mistake_window=500
):
    """
    Method to compute eviction mistake rate.
    :param metrics_logger: The metrics logger.
    :param mistake_window: The mistake window to consider.
    :return: The eviction mistake rate.
    """
    # initial message
    info("🔄 Eviction mistake rate calculation started...")

    try:
        # initialize data
        mistakes = 0
        total_eviction_events = 0

        # count mistakes within a temporal window
        for key, eviction_times in metrics_logger.evicted_keys.items():
            for eviction_time in eviction_times:
                total_eviction_events += 1
                # look for any future access after the current eviction
                future_accesses = [
                    t for t in metrics_logger.access_events.get(key, [])
                    if t > eviction_time and (t - eviction_time) <= mistake_window
                ]
                if future_accesses:
                    mistakes += 1
    except (
        AttributeError,
        TypeError,
        ZeroDivisionError
    ) as e:
        raise RuntimeError(f"❌ Error while computing eviction mistake rate: {e}.")

    # show a successful message
    info("🟢 Eviction mistake rate computed.")

    return mistakes / total_eviction_events if total_eviction_events > 0 else 0


def compute_prefetch_hit_rate(
        prefetch_hits,
        tot_prefetch,
):
    """
    Method to compute prefetch hit rate.
    :param prefetch_hits: The number of prefetch hits.
    :param tot_prefetch: Total number of prefetches.
    :return: The prefetch hit rate.
    """
    # initial message
    info("🔄 Prefetch hit rate calculation started...")

    try:
        if tot_prefetch > 0:
            # calculate prefetch hit rate
            prefetch_hit_rate =  prefetch_hits / tot_prefetch
        else:
            prefetch_hit_rate = 0.0

    except (
        AttributeError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        NameError
    ) as e:
        raise RuntimeError(f"❌ Error while computing prefetch hit rate: {e}.")

    # show a successful message
    info("🟢 Prefetch hit rate computed.")

    return prefetch_hit_rate


def compute_ttl_hit_ratio(metrics_logger):
    """
    Method to compute the TTL Hit Ratio.
    :param metrics_logger: The metrics logger.
    :return: The TTL Hit Ratio or None if no puts.
    """
    # initial message
    info("🔄 TTL Hit Ratio calculation started...")

    # initialize counters
    hits_before_expiry = 0
    total_accesses = 0

    try:
        for key, puts in metrics_logger.put_events.items():
            # retrieve key accesses
            accesses = sorted(metrics_logger.access_events.get(key, []))
            used_access_indices = set()

            # calculate expiration time for the current put
            for put_time, ttl in puts:
                expiry_time = put_time + ttl

                # for each key access
                for i, access_time in enumerate(accesses):
                    # check if the access occur while TTL was valid
                    if i in used_access_indices:
                        continue
                    if put_time <= access_time <= expiry_time:
                        hits_before_expiry += 1
                        used_access_indices.add(i)
                        break

            total_accesses += len(accesses)

        # success message
        info("🟢 TTL Hit Ratio computed.")
        return hits_before_expiry / total_accesses

    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        NameError
    ) as e:
        raise RuntimeError(f"❌ Error while computing TTL Hit Ratio: {e}.")


def calculate_hit_miss_rate(counters):
    """
    Method to calculate hit and miss rate.
    :param counters: A counter used while simulating a cache policy
    :return: The hit and miss rate in terms of %.
    """
    # initial message
    info("🔄 Hit and miss rate calculation started...")

    try:
        # calculate hit rate and miss rate in terms of %
        total = counters['hits'] + counters['misses']
        hit_rate = counters['hits'] / total * 100
        miss_rate = counters['misses'] / total * 100
    except (
            KeyError,
            ZeroDivisionError,
            TypeError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error while calculating hit and miss rate: {e}.")

    # show a successful message
    info("🟢 Hit and miss rate calculated.")

    return hit_rate, miss_rate


def calculate_prefetching_avg_latency(autoregressive_latencies):
    """
    Method to calculate prefetching average latency.
    :return: Prefetching average latency.
    """
    # initial message
    info("🔄 Prefetching average latency calculation started...")

    try:
        avg_prefetching_latency = 0
        if (
            autoregressive_latencies is not None and
            not len(autoregressive_latencies) == 0
        ):
            # filter only not None elements
            valid_latencies = [
                lat for lat in autoregressive_latencies
                if lat is not None
            ]
            if valid_latencies:
                # calculate average prefetching latency
                avg_prefetching_latency = (
                        sum(valid_latencies)/
                        len(valid_latencies)
                )
    except (
        TypeError,
        ZeroDivisionError,
        AttributeError,
        NameError,
        ValueError,
    ) as e:
        raise RuntimeError(f"❌ Error while calculating average prefetching latency: {e}.")

    # show a successful message
    info("🟢 Prefetching average latency calculated.")

    return avg_prefetching_latency


def compute_cache_metrics(
        counters,
        tot_prefetch,
        autoregressive_latencies,
        metrics_logger
):
    """
    Method to orchestrate cache metrics calculation.
    :param counters: A counter used while simulating a cache policy.
    :param tot_prefetch: The total number of prefetches.
    :param autoregressive_latencies: The autoregressive latencies.
    :param metrics_logger: The metrics logger.
    :return: All the computed cache metrics.
    """
    # initial message
    info("🔄 Cache metrics calculation started...")

    # calculate hit rate and miss rate
    (
        hit_rate,
        miss_rate
    ) = calculate_hit_miss_rate(
        counters
    )

    # component evaluation
    prefetch_hit_rate = compute_prefetch_hit_rate(
        counters['hits'] - counters['hits_cold_start'],
        tot_prefetch
    )
    ttl_hit_ratio = compute_ttl_hit_ratio(
        metrics_logger
    )
    eviction_mistake_rate = compute_eviction_mistake_rate(
        metrics_logger
    )

    # calculate avg prefetching average
    avg_prefetching_latency = calculate_prefetching_avg_latency(
        autoregressive_latencies
    )

    # show a successful message
    info("🟢 Cache metrics calculated.")

    return (
        hit_rate,
        miss_rate,
        prefetch_hit_rate,
        ttl_hit_ratio,
        eviction_mistake_rate,
        avg_prefetching_latency
    )