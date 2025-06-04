from collections import defaultdict
from utils.logs.log_utils import info


class CacheMetricsLogger:

    def __init__(self):
        """
        Method to initialize the cache metrics logger class.
        """
        # initial message
        info("🔄 CacheMetricsLogger initialization started...")

        try:
            self.put_events = {}
            self.access_events = defaultdict(list)
            self.evicted_keys = defaultdict(list)
            self.prefetch_predictions = {}
        except (
            NameError,
            TypeError,
            AttributeError
        ) as e:
            raise RuntimeError(f"❌ Error while instantiating cache metrics logger: {e}.")

        # print a successful message
        info("🟢 CacheMetricsLogger initialized.")

    def log_put(
            self,
            key,
            time,
            ttl
    ):
        """
        Method to trace keys inserted into the cache
        :param key: The key inserted into the cache.
        :param time: The time the key was inserted.
        :param ttl: The TTL of the key.
        :return:
        """
        # initial message
        info("🔄 Key insertion tracing started...")

        try:
            self.put_events.setdefault(key, []).append((time, ttl))
        except (
            NameError,
            TypeError,
            AttributeError
        ) as e:
            raise RuntimeError(f"❌ Error while keeping track of put events: {e}.")

        # print a successful message
        info("🟢 Key insertion traced.")


    def log_get(
            self,
            key,
            time
    ):
        """
        Method to trace key accesses from the cache.
        :param key: The key accesses from the cache.
        :param time: The time the key was accessed.
        :return:
        """
        # initial message
        info("🔄 Key access tracing started...")

        try:
            self.access_events[key].append(time)
        except (
            AttributeError,
            NameError,
            TypeError
        ) as e:
            raise RuntimeError(f"❌ Error while keeping track of access events: {e}.")

        # print a successful message
        info("🟢 Key access traced.")


    def log_eviction(
            self,
            key,
            time
    ):
        """
        Method to trace key evictions from the cache.
        :param key: The key evicted from the cache.
        :param time: The time the key was evicted.
        :return:
        """
        # initial message
        info("🔄 Key eviction tracing started...")

        try:
            self.evicted_keys[key].append(time)
        except (
            AttributeError,
            NameError,
            TypeError
        ) as e:
            raise RuntimeError(f"❌ Error while keeping track of evictions: {e}.")

        # print a successful message
        info("🟢 Key eviction traced.")


    def log_prefetch_prediction(
            self,
            time,
            predicted_keys
    ):
        """
        Method to trace keys prefetching.
        :param time: The time the key was prefetched.
        :param predicted_keys: The predicted key.
        :return:
        """
        # initial message
        info("🔄 Key prefetching tracing started...")

        try:
            self.prefetch_predictions[time] = predicted_keys
        except (
            AttributeError,
            TypeError,
            NameError
        ) as e:
            raise RuntimeError(f"❌ Error while keeping track of prefetched predictions: {e}.")

        # print a successful message
        info("🟢 Key prefetching traced.")