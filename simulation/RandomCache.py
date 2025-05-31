import random
from utils.log_utils import debug


class RandomCache:

    def __init__(self, config_settings):
        """
        Method to initialize a random cache of size maxsize.
        :param config_settings: The configuration settings.
        """
        # initialize data
        self.maxsize = config_settings.cache_size
        self.store = {}
        self.expiry = {}

        # debugging
        debug(f"⚙️Max cache size: {self.maxsize}.")


    def contains(self, key, current_time):
        """
        Method to check if a key is in the cache.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is in the cache and
        is not expired, False otherwise.
        """
        try:
            is_contained = (
                    key in self.store and not
                    self._is_expired(key, current_time)
            )
        except (AttributeError, TypeError, KeyError) as e:
            raise RuntimeError(f"❌ Error while checking if the key "
                               f"is contained into the cache: {e}.")

        # debugging
        debug(f"⚙️Key: {key}, contained: {is_contained}.")

        return is_contained


    def get(self, key, current_time):
        """
        Method to get a key from the cache.
        :param key: The key to get.
        :param current_time: The current time.
        :return: The key from the cache if present, None otherwise.
        """
        try:
            if self.contains(key, current_time):
                # debugging
                debug(f"⚙️Key accessed (time: {current_time}): {key}.")
                return self.store[key]
            return None
        except (AttributeError, KeyError, TypeError) as e:
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
            self.store.pop(k, None)
            self.expiry.pop(k, None)


    def put(self, key, ttl, current_time):
        """
        Method to put a key in the cache.
        :param key: The key to put.
        :param ttl: The TTL of the key.
        :param current_time: The current time.
        :return:
        """
        try:
            # clean up the cache
            self._remove_expired_keys(current_time)

            # check if the key is in the cache
            if self.contains(key, current_time):
                # update the expiration time of the key
                self.expiry[key] = current_time + ttl

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

            # store the key
            self.store[key] = key
            self.expiry[key] = current_time + ttl

            # debugging
            debug(f"⚙️Key {key} put in the cache with TTL: {self.expiry[key]}.")

        except (AttributeError, TypeError, KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"❌ Error while putting the key"
                               f" into the cache: {e}.")


    def _is_expired(self, key, current_time):
        """
        Method to check if a key is expired.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is expired, False otherwise.
        """
        try:
            return self.expiry.get(key, 0) < current_time
        except (AttributeError, TypeError) as e:
            raise RuntimeError(f"❌ Error while checking if the "
                               f"key is expired: {e}.")