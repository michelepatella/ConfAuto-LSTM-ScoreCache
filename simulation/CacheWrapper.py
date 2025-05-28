class CacheWrapper:

    def __init__(self, cache_class, config_settings):
        """
        Method to initialize a CacheWrapper, used
        to manage baseline caching strategies.
        :param cache_class: The type of the cache.
        :param config_settings: The configuration settings.
        :return:
        """
        self.cache = cache_class(
            maxsize=config_settings.cache_size
        )
        self.expiry = {}


    def _is_expired(self, key, current_time):
        """
        Method to check if a key is expired.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is expired, False otherwise.
        """
        return (
                key in self.expiry and
                self.expiry[key] < current_time
        )


    def contains(self, key, current_time):
        """
        Method to check if a key is contained into the cache.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is contained into the cache, False otherwise.
        """
        # check if the key is in the cache
        if (
            key in self.cache and
            not self._is_expired(key, current_time)
        ):
            return True
        else:
            # if the key is expired, it's going to be removed
            if key in self.cache:
                self.cache.pop(key, None)
                self.expiry.pop(key, None)
            return False


    def get(self, key, current_time):
        """
        Method to get a key from the cache.
        :param key: The key to get.
        :param current_time: The current time.
        :return: The key from the cache if present, None otherwise.
        """
        # check if the key is in the cache
        if self.contains(key, current_time):
            return self.cache[key]
        else:
            return None


    def put(self, key, ttl, current_time):
        """
        Method to put a key into the cache.
        :param key: The key to put.
        :param ttl: The TTL for the key.
        :param current_time: The current time.
        :return:
        """
        # set the key into the cache
        self.cache[key] = key
        # set expiration time for the key
        self.expiry[key] = current_time + ttl