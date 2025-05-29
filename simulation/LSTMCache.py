import math
from utils.log_utils import debug


class LSTMCache:

    def __init__(self, config_settings):
        """
        Method to initialize the LSTM cache.
        :param config_settings: The configuration settings.
        :return:
        """
        # initialize data
        self.store = {}
        self.expiry = {}
        self.scores = {}
        self.maxsize = config_settings.cache_size
        self.threshold_score = config_settings.threshold_score
        self.ttl_base = config_settings.ttl_base

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


    def put(
            self,
            key,
            score,
            current_time
    ):
        """
        Method to put a key into the cache, containing the full
        logic of prefetching, TTL assigment, and eviction.
        :param key: The key to put.
        :param score: The score associated with the key.
        :param current_time: The current time.
        :return:
        """
        # check if the key must be removed
        # in case its score is less than the threshold
        try:

            if score < self.threshold_score:
                # debugging
                debug(f"⚙️Key: {key}, Score: {score}.")

                # remove the key from the cache
                self.store.pop(key, None)
                self.expiry.pop(key, None)
                self.scores.pop(key, None)
                return

            # compute TTL dynamically
            ttl = self.ttl_base * math.log1p(score * 10)
            # use the TTL base in case the dynamic TTL <= 0
            if ttl <= 0:
                ttl = self.ttl_base

            # debugging
            debug(f"⚙️Key: {key}, Dynamic TTL: {ttl}.")

            # if the key is new and the cache is full, evict something
            if (
                key not in self.store and
                len(self.store) >= self.maxsize
            ):
                # identify key candidates to be evicted
                # as the ones whose score is less than the threshold
                evict_candidates = [
                    k for k in self.store
                    if self.scores.get(k, 0) < self.threshold_score
                ]

                # debugging
                debug(f"⚙️Full cache, evicting: {evict_candidates}.")

                # evict all candidate keys
                for k in evict_candidates:
                    self.store.pop(k, None)
                    self.expiry.pop(k, None)
                    self.scores.pop(k, None)

                # check if the cache is still full
                while len(self.store) >= self.maxsize:
                    # evict the key with the lowest score
                    key_to_evict = min(
                        self.store.keys(),
                        key=lambda k: self.scores.get(k, 0)
                    )

                    # debugging
                    debug(f"⚙️Cache still full, evicting: {key_to_evict}.")

                    self.store.pop(key_to_evict, None)
                    self.expiry.pop(key_to_evict, None)
                    self.scores.pop(key_to_evict, None)

            # put the key
            self.store[key] = key
            self.expiry[key] = current_time + ttl
            self.scores[key] = score

            # debugging
            debug(f"⚙️Key {key} put in the cache with TTL: {self.expiry[key]}.")

        except (AttributeError, TypeError, KeyError, ValueError) as e:
            raise RuntimeError(f"❌ Error while putting the key"
                               f" into the cache: {e}.")