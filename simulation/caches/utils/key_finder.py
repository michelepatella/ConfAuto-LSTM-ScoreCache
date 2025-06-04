from utils.logs.log_utils import info


def find_key(
        cache,
        key,
        current_time,
        counters
):
    """
    Method to find a key in the cache.
    :param cache: The cache where to search the key.
    :param key: The key to search.
    :param current_time: The current time.
    :param counters: Counters used during the simulation.
    :return:
    """
    # initial message
    info(f"🔄 Key search started...")

    try:
        # check if the cache contains the key
        if cache.contains(
                key,
                current_time
        ):
            # increment cache hits
            counters['hits'] += 1
            info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | HIT")
            # print a successful message
            info(f"🟢 Key search completed.")
            return True
        else:
            # increment cache misses
            counters['misses'] += 1
            info(f"ℹ️ Time: {current_time:.2f} | Key: {key} | MISS")
            # print a successful message
            info(f"🟢 Key search completed.")
            return False
    except (
        NameError,
        TypeError,
        AttributeError,
        ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while searching a key into the cache: {e}.")