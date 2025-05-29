from utils.log_utils import info


def handle_traditional_cache_policy(
        cache,
        policy_name,
        key,
        current_time,
        counters,
        config_settings
):
    """
    Method to handle traditional cache policies.
    :param cache: The cache.
    :param policy_name: The policy name.
    :param key: The current key.
    :param current_time: The current time.
    :param counters: Hits and misses counters.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info(f"🔄 {policy_name} policy management started...")

    try:
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

        # the key should be always put in the random cache
        if policy_name == "RANDOM":
            # put the key in the cache
            cache.put(
                key,
                config_settings.ttl_base,
                current_time
            )

    except (AttributeError, KeyError, TypeError, NameError, ValueError) as e:
        raise RuntimeError(f"❌ Error while handling {policy_name} policy: {e}.")

    # print a successful message
    info(f"🟢 {policy_name} policy management completed.")