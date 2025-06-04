from collections import defaultdict

from cachetools import Cache


class LFUCache(Cache):

    def __init__(
            self,
            maxsize,
            callback=None
    ):
        """
        Method to instantiate a LFU cache.
        :param maxsize: Max size of the LFU cache.
        :param callback: The callback object.
        """
        super().__init__(maxsize)
        self.data = {}
        self.freq = defaultdict(int)
        self.callback = callback


    def __getitem__(
            self,
            key
    ):
        """
        Method to retrieve a key from the LFU cache.
        :param key: The key to retrieve.
        :return: The retrieved key.
        """
        self.freq[key] += 1
        return self.data[key]


    def __setitem__(
            self,
            key,
            value
    ):
        """
        Method to store a value in the LFU cache.
        :param key: The key to store.
        :param value: The value to store.
        :return:
        """
        if key in self.data:
            self.data[key] = value
            self.freq[key] += 1
        else:
            if len(self.data) >= self.maxsize:
                # Find LFU key(s)
                min_freq = min(self.freq.values())
                candidates = [k for k, f in self.freq.items() if f == min_freq]
                # Evict the oldest among them
                key_to_evict = candidates[0]
                del self.data[key_to_evict]
                del self.freq[key_to_evict]
                if self.callback:
                    self.callback(key_to_evict)
            self.data[key] = value
            self.freq[key] = 1


    def __delitem__(
            self,
            key
    ):
        """
        Method to remove a key from the LFU cache.
        :param key: The key to remove.
        :return:
        """
        del self.data[key]
        del self.freq[key]


    def __contains__(
            self,
            key
    ):
        """
        Method to check if a key exists in the LFU cache.
        :param key: The key to search.
        :return: True if the key exists in the LFU cache, False otherwise.
        """
        return key in self.data


    def pop(
            self,
            key,
            default=None
    ):
        """
        Method to remove and return the key from the LFU cache.
        :param key: The key to remove.
        :param default: Default value to return if the key does not exist.
        :return: The retrieved key.
        """
        if key in self.data:
            value = self.data.pop(key)
            self.freq.pop(
                key,
                None
            )
            return value
        return default


    def __len__(self):
        """
        Method to return the length of the LFU cache.
        :return: The length of the LFU cache.
        """
        return len(self.data)


    def clear(self):
        """
        Method to clear the LFU cache.
        :return:
        """
        self.data.clear()
        self.freq.clear()
