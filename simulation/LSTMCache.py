import math
import random

from simulation.CacheMetricsLogger import CacheMetricsLogger
from utils.log_utils import debug


class LSTMCache:

    def __init__(
            self,
            metrics_logger,
            config_settings
    ):
        """
        Method to initialize the LSTM cache.
        :param config_settings: The configuration settings.
        :param metrics_logger: The metrics logger.
        :return:
        """
        # initialize data
        self.store = {}
        self.expiry = {}
        self.scores = {}
        self.maxsize = config_settings.cache_size
        self.threshold_score = config_settings.threshold_score
        self.ttl_base = config_settings.ttl_base
        self.metrics_logger = metrics_logger

        # debugging
        debug(f"⚙️Max cache size: {self.maxsize}.")
        debug(f"⚙️Threshold score: {self.threshold_score}.")
        debug(f"⚙️TTL base: {self.ttl_base}.")


    def contains(self, key, current_time):
        """
        Method to check if a key is in the cache.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is in the cache, False otherwise.
        """
        try:
            # trace event
            self.metrics_logger.log_get(key, current_time)

            is_contained = (
                    key in self.store and
                    self.expiry.get(key, 0) > current_time
            )
        except (AttributeError, TypeError) as e:
            raise RuntimeError(f"❌ Error while checking if the key "
                               f"is contained into the cache: {e}.")

        # debugging
        debug(f"⚙️Key: {key}, contained: {is_contained}.")

        return is_contained


    def get(self, key, current_time):
        """
        Method to get the key from the cache.
        :param key: The key to get.
        :param current_time: The current time.
        :return: The key from the cache if present, None otherwise.
        """
        try:
            # check if the key is in the cache
            if self.contains(key, current_time):
                # debugging
                debug(f"⚙️Key accessed (time: {current_time}): {key}.")
                return self.store[key]
            return None
        except (AttributeError,TypeError, KeyError) as e:
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
            expiry_time in self.expiry.items()
            if expiry_time <= current_time
        ]
        # remove expired keys
        for k in expired_keys:
            self.store.pop(k, None)
            self.expiry.pop(k, None)
            self.scores.pop(k, None)
            # trace event
            self.metrics_logger.log_eviction(k, current_time)


    def put(
            self,
            key,
            score,
            current_time,
            cold_start=False,
            config_settings=None
    ):
        """
        Method to put a key into the cache, containing the full
        logic of prefetching, TTL assigment, and eviction.
        :param key: The key to put.
        :param score: The score associated with the key.
        :param current_time: The current time.
        :param cold_start: Specifies whether the key is cold or not.
        :param config_settings: The configuration settings.
        :return:
        """
        # check if the key must be removed
        # in case its score is less than the threshold
        try:

            # clean up the cache
            self._remove_expired_keys(current_time)

            # if the cold-start is overcome
            if not cold_start:
                if score < self.threshold_score:
                    # debugging
                    debug(f"⚙️Key: {key}, Score: {score}.")

                    # remove the key from the cache
                    self.store.pop(key, None)
                    self.expiry.pop(key, None)
                    self.scores.pop(key, None)

                    # trace event
                    self.metrics_logger.log_eviction(key, current_time)

                    return

                # compute TTL dynamically
                ttl = self.ttl_base * (1 + math.log1p(score))

                # debugging
                debug(f"⚙️Key: {key}, Dynamic TTL: {ttl}.")

                # if the key is new and the cache is full, evict something
                if (
                    key not in self.store and
                    len(self.store) >= self.maxsize
                ):

                    # evict the key with the lowest score
                    key_to_evict = min(
                        self.store.keys(),
                        key=lambda k: self.scores.get(k, 0)
                    )

                    # debugging
                    debug(f"⚙️Cache still full, evicting: {key_to_evict}.")

                    # evict the key if its score is less than
                    # the one of the key to be inserted
                    if self.scores.get(key_to_evict, 0) <= score:
                        self.store.pop(key_to_evict, None)
                        self.expiry.pop(key_to_evict, None)
                        self.scores.pop(key_to_evict, None)

                        # trace event
                        self.metrics_logger.log_eviction(key_to_evict, current_time)
                    else:
                        return

                # put the key
                self.store[key] = key
                self.expiry[key] = current_time + ttl
                self.scores[key] = score

                # trace event
                self.metrics_logger.log_prefetch_prediction(current_time, [key])

                # trace event
                self.metrics_logger.log_put(key, current_time, ttl)

            else:
                # cold-start management

                # check if the key is in the cache
                if self.contains(key, current_time):
                    # update the expiration time of the key
                    self.expiry[key] = current_time + config_settings.fixed_ttl

                    # trace event
                    self.metrics_logger.log_put(key, current_time, config_settings.fixed_ttl)

                    # debugging
                    debug(f"⚙️ Key {key} already in the cache, new TTL: {self.expiry[key]}.")
                    return

                # check if the cache is full
                elif len(self.store) >= self.maxsize:
                    # evict a key randomly
                    evict_key = random.choice(
                        list(self.store.keys())
                    )

                    # debugging
                    debug(f"⚙️Full cache, evicting: {evict_key}.")

                    self.store.pop(evict_key)
                    self.expiry.pop(evict_key)

                    # trace event
                    self.metrics_logger.log_eviction(evict_key, current_time)

                # store the key
                self.store[key] = key
                self.scores[key] = score
                self.expiry[key] = current_time + config_settings.fixed_ttl

                # trace event
                self.metrics_logger.log_put(key, current_time, config_settings.fixed_ttl)

            # debugging
            debug(f"⚙️Key {key} put in the cache with TTL: {self.expiry[key]}.")

        except (AttributeError, TypeError, KeyError, ValueError) as e:
            raise RuntimeError(f"❌ Error while putting the key"
                               f" into the cache: {e}.")