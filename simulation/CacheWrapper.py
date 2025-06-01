from simulation.CacheMetricsLogger import CacheMetricsLogger
from utils.log_utils import debug


class CacheWrapper:

    def __init__(
            self,
            cache_class,
            metrics_logger,
            config_settings
    ):
        """
        Method to initialize a CacheWrapper, used
        to manage baseline caching strategies (except Random).
        :param cache_class: The cache object.
        :param config_settings: The configuration settings.
        :param metrics_logger: The metrics logger.
        :return:
        """
        self.cache = cache_class(
            maxsize=config_settings.cache_size
        )
        self.expiry = {}
        self.metrics_logger = metrics_logger

        # debugging
        debug(f"⚙️ Max cache size: {self.cache.maxsize}.")


    def _is_expired(self, key, current_time):
        """
        Method to check if a key is expired.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is expired, False otherwise.
        """
        try:
            return (
                    key in self.expiry and
                    self.expiry[key] < current_time
            )
        except (AttributeError, TypeError, KeyError) as e:
            raise RuntimeError(f"❌ Error while checking if the "
                               f"key is expired: {e}.")


    def contains(self, key, current_time):
        """
        Method to check if a key is contained into the cache.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is contained into the cache, False otherwise.
        """
        try:
            # check if the key is in the cache
            if (
                key in self.cache and
                not self._is_expired(key, current_time)
            ):
                # debugging
                debug(f"⚙️Key: {key}, contained: True.")

                return True
            else:
                # if the key is expired, it's going to be removed
                if key in self.cache:

                    # debugging
                    debug(f"⚙️Key {key} expired, it's going to be removed.")

                    self.cache.pop(key, None)
                    self.expiry.pop(key, None)

                    # trace event
                    self.metrics_logger.log_eviction(key, current_time)
                return False
        except (AttributeError, TypeError, KeyError) as e:
            raise RuntimeError(f"❌ Error while checking if the key "
                               f"is contained into the cache: {e}.")


    def get(self, key, current_time):
        """
        Method to get a key from the cache.
        :param key: The key to get.
        :param current_time: The current time.
        :return: The key from the cache if present, None otherwise.
        """
        try:
            # trace event
            self.metrics_logger.log_get(key, current_time)

            # check if the key is in the cache
            if self.contains(key, current_time):
                # debugging
                debug(f"⚙️ Key accessed (time: {current_time}): {key}.")

                return self.cache[key]
            else:
                return None
        except (AttributeError, TypeError, KeyError) as e:
            raise RuntimeError(f"❌ Error while getting the key"
                               f" from the cache: {e}.")


    def _remove_expired_keys(self, current_time):
        """
        Method to remove expired keys from the cache.
        :param current_time: The current time.
        :return:
        """
        # identify expired keys
        expired_keys = [
            k for k,
            exp_time in self.expiry.items()
            if exp_time < current_time
        ]
        # remove expired keys
        for k in expired_keys:
            self.cache.pop(k, None)
            self.expiry.pop(k, None)

            # trace event
            self.metrics_logger.log_eviction(k, current_time)


    def put(self, key, ttl, current_time):
        """
        Method to put a key into the cache.
        :param key: The key to put.
        :param ttl: The TTL for the key.
        :param current_time: The current time.
        :return:
        """
        try:
            # trace event
            self.metrics_logger.log_put(key, current_time, ttl)

            # clean up the cache
            self._remove_expired_keys(current_time)

            # set the key into the cache
            self.cache[key] = key
            # set expiration time for the key
            self.expiry[key] = current_time + ttl

            # debugging
            debug(f"⚙️Key {key} put in the cache with TTL: {self.expiry[key]}.")
        except (AttributeError, TypeError) as e:
            raise RuntimeError(f"❌ Error while putting the key"
                               f" into the cache: {e}.")