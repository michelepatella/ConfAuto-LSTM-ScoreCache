from collections import OrderedDict
from cachetools import Cache


class FIFOCache(Cache):

    def __init__(
            self,
            maxsize,
            callback=None
    ):
        """
        Method to instantiate a FIFO cache.
        :param maxsize: Max size of the FIFO cache.
        :param callback: The callback object
        """
        super().__init__(maxsize)
        self._data = OrderedDict()
        self.callback = callback


    def __getitem__(
            self,
            key
    ):
        """
        Method to retrieve a key from the FIFO cache.
        :param key: The key to retrieve.
        :return: The retrieved key.
        """
        return self._data[key]


    def __setitem__(
            self,
            key,
            value
    ):
        """
        Method to store a value in the FIFO cache.
        :param key: The key to store.
        :param value: The value to store.
        :return:
        """
        if key in self._data:
            self._data[key] = value
        else:
            if len(self._data) >= self.maxsize:
                old_key, _ = self._data.popitem(last=False)
                if self.callback:
                    self.callback(old_key)
            self._data[key] = value


    def __delitem__(
            self,
            key
    ):
        """
        Method to remove a key from the FIFO cache.
        :param key: The key to remove.
        :return:
        """
        del self._data[key]


    def __contains__(
            self,
            key
    ):
        """
        Method to check if a key exists in the FIFO cache.
        :param key: The key to search.
        :return: True if the key exists in the FIFO cache, False otherwise.
        """
        return key in self._data


    def pop(
            self,
            key,
            default=None
    ):
        """
        Method to remove and return the key from the FIFO cache.
        :param key: The key to remove.
        :param default: Default value to return if the key does not exist.
        :return: The retrieved key.
        """
        return self._data.pop(
            key,
            default
        )


    def __len__(self):
        """
        Method to return the length of the FIFO cache.
        :return: The length of the FIFO cache.
        """
        return len(self._data)


    def clear(self):
        """
        Method to clear the FIFO cache.
        :return:
        """
        self._data.clear()