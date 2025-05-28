import random


class RandomCache:

    def __init__(self, maxsize):
        """
        Method to initialize a random cache of size maxsize.
        :param maxsize: The maximum size of the cache.
        """
        self.maxsize = maxsize
        self.store = {}
        self.expiry = {}


    def contains(self, key, current_time):
        """
        Method to check if a key is in the cache.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is in the cache, False otherwise.
        """
        return (
                key in self.store and not
                self._is_expired(key, current_time)
        )


    def get(self, key, current_time):
        """
        Method to get a key from the cache.
        :param key: The key to get.
        :param current_time: The current time.
        :return: The key from the cache if present, None otherwise.
        """
        if self.contains(key, current_time):
            return self.store[key]
        return None


    def put(self, key, ttl, current_time):
        """
        Method to put a key in the cache.
        :param key: The key to put.
        :param ttl: The TTL of the key.
        :param current_time: The current time.
        :return:
        """
        # check if the key is in the cache
        if self.contains(key, current_time):
            # update the expiration time of the key
            self.expiry[key] = current_time + ttl
        # check if the cache is full
        elif len(self.store) >= self.maxsize:
            # evict a key randomly
            evict_key = random.choice(
                list(self.store.keys())
            )
            self.store.pop(evict_key)
            self.expiry.pop(evict_key)
        # store the key
        self.store[key] = key
        self.expiry[key] = current_time + ttl


    def _is_expired(self, key, current_time):
        """
        Method to check if a key is expired.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is expired, False otherwise.
        """
        return self.expiry.get(key, 0) < current_time