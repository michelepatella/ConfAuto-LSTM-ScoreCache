from abc import ABC, abstractmethod
from utils.log_utils import debug


class BaseCache(ABC):

    def __init__(
            self,
            cache_class,
            metrics_logger,
            config_settings
    ):
        """
        Method to initialize the base cache class.
        :param cache_class: The cache class.
        :param metrics_logger: The metrics logger object.
        :param config_settings: The configuration settings.
        """
        # initialize data
        try:
            if cache_class is not None:
                self.cache = cache_class(
                    config_settings.cache_size
                )
            else:
                self.cache = None
            self.maxsize = config_settings.cache_size
            self.store = {}
            self.expiry = {}
            self.scores = {}
            self.metrics_logger = metrics_logger
        except (
            AttributeError,
            TypeError,
            NameError,
            ValueError
        ) as e:
            raise RuntimeError(f"❌ Error while initializing the base cache: {e}.")


    def _is_expired(
            self,
            key,
            current_time
    ):
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
        except (
                AttributeError,
                TypeError,
                KeyError
        ) as e:
            raise RuntimeError(f"❌ Error while checking if the key is expired: {e}.")


    def _remove_expired_keys(
            self,
            current_time
    ):
        """
        Method to remove expired keys from the cache.
        :param current_time: The current time.
        :return:
        """
        try:
            # identify expired keys
            expired_keys = [
                k for k,
                exp_time in self.expiry.items()
                if exp_time < current_time
            ]

            # remove expired keys
            for k in expired_keys:
                if self.cache is not None:
                    self.cache.pop(k, None)
                else:
                    self.store.pop(k, None)
                self.expiry.pop(k, None)

                if self.scores is not None:
                    self.scores.pop(k, None)

                # trace event
                self.metrics_logger.log_eviction(
                    k,
                    current_time
                )
        except (
            AttributeError,
            TypeError,
            NameError,
            KeyError,
            ValueError
        ) as e:
            raise RuntimeError(f"❌ Error while removing expired keys from cache: {e}.")


    def contains(
            self,
            key,
            current_time
    ):
        """
        Method to check if a key is in the cache.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is in the cache and
        is not expired, False otherwise.
        """
        try:
            # check if the key is in the cache
            if (
                self.cache is not None and
                key in self.cache and not
                self._is_expired(
                    key,
                    current_time
                )
            ):
                # debugging
                debug(f"⚙️Key: {key} contained in cache.")
                return True
            elif (
                key in self.store and not
                self._is_expired(
                    key,
                    current_time
                )
            ):
                # debugging
                debug(f"⚙️Key: {key} not contained in cache.")
                return True
            else:
                return False
        except (
                AttributeError,
                TypeError,
                KeyError
        ) as e:
            raise RuntimeError(f"❌ Error while checking if the key is cached: {e}.")


    def get(
            self,
            key,
            current_time
    ):
        """
        Method to get a key from the cache.
        :param key: The key to get.
        :param current_time: The current time.
        :return: The key from the cache if present, None otherwise.
        """
        try:
            # trace event
            self.metrics_logger.log_get(
                key,
                current_time
            )

            # check if the key is in the cache
            if self.contains(
                    key,
                    current_time
            ):
                # debugging
                debug(f"⚙️ Key accessed (time: {current_time}): {key}.")

                if self.cache is not None:
                    return self.cache[key]
                else:
                    return self.store[key]
            else:
                return None
        except (
                AttributeError,
                TypeError,
                KeyError
        ) as e:
            raise RuntimeError(f"❌ Error while getting the key from the cache: {e}.")


    @abstractmethod
    def put(self, *args, **kwargs):
        """
        Each strategy defines its own put method.
        :param self:
        :param args:
        :param kwargs:
        :return:
        """