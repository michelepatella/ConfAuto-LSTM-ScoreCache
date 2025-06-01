from collections import defaultdict

class CacheMetricsLogger:

    def __init__(self):
        """
        Method to initialize the cache metrics logger class.
        """
        self.put_events = {}
        self.access_events = defaultdict(list)
        self.evicted_keys = {}
        self.prefetch_predictions = {}


    def log_put(self, key, time, ttl):
        """
        Method to trace keys inserted into the cache
        :param key: The key inserted into the cache.
        :param time: The time the key was inserted.
        :param ttl: The TTL of the key.
        :return:
        """
        self.put_events[key] = (time, ttl)


    def log_get(self, key, time):
        """
        Method to trace key accesses from the cache.
        :param key: The key accesses from the cache.
        :param time: The time the key was accessed.
        :return:
        """
        self.access_events[key].append(time)


    def log_eviction(self, key, time):
        """
        Method to trace key evictions from the cache.
        :param key: The key evicted from the cache.
        :param time: The time the key was evicted.
        :return:
        """
        self.evicted_keys[key] = time


    def log_prefetch_prediction(self, time, predicted_keys):
        """
        Method to trace keys prefetching.
        :param time: The time the key was prefetched.
        :param predicted_keys: The predicted key.
        :return:
        """
        self.prefetch_predictions[time] = predicted_keys