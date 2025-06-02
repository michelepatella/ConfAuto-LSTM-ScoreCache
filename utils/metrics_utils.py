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


def compute_eviction_mistake_rate(metrics_logger):
    """
    Method to compute eviction mistake rate.
    :param metrics_logger: The metrics logger.
    :return: The eviction mistake rate.
    """
    # initial message
    info("🔄 Eviction mistake rate calculation started...")

    try:
        # initialize data
        mistakes = 0
        total = len(metrics_logger.evicted_keys)

        # count mistakes due to wrongly evictions
        for key, eviction_time in metrics_logger.evicted_keys.items():
            future_accesses = [
                t for t in metrics_logger.access_events.get(key, [])
                if t > eviction_time
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

    return mistakes / total if total > 0 else 0


def compute_prefetch_hit_rate(
        metrics_logger,
        window_size
):
    """
    Method to compute prefetch hit rate.
    :param metrics_logger: The metrics logger.
    :param window_size: The window size.
    :return: The prefetch hit rate.
    """
    # initial message
    info("🔄 Prefetch hit rate calculation started...")

    # initialize data
    hits = 0
    total = 0

    try:
        # count prefetched keys have been hit
        for t, predicted_keys in metrics_logger.prefetch_predictions.items():
            if not isinstance(
                    predicted_keys,
                    (list, set, tuple)
            ):
                predicted_keys = [predicted_keys]
            total += len(predicted_keys)
            for key in predicted_keys:
                for access_time in metrics_logger.access_events.get(key, []):
                    if t < access_time <= t + window_size:
                        hits += 1
                        break
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

    return hits / total if total > 0 else 0


def compute_ttl_mae(metrics_logger):
    """
    Method to compute TTL MAE.
    :param metrics_logger: The metrics logger.
    :return: The MAE.
    """
    # initial message
    info("🔄 TTL MAE calculation started...")

    errors = []

    try:
        # calculate MAE on TTL assigned
        for key, (put_time, predicted_ttl) in metrics_logger.put_events.items():
            actual_accesses = metrics_logger.access_events.get(key, [])
            if actual_accesses:
                last_use = max([
                    t for t in actual_accesses
                    if t >= put_time],
                    default=None
                )
                if last_use:
                    true_ttl = last_use - put_time
                    errors.append(abs(true_ttl - predicted_ttl))
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        NameError
    ) as e:
        raise RuntimeError(f"❌ Error while computing TTL MAE: {e}.")

    # show a successful message
    info("🟢 TTL MAE computed.")

    return np.mean(errors) if errors else None


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