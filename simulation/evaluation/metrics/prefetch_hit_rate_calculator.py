from utils.logs.log_utils import info


def calculate_prefetch_hit_rate(
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