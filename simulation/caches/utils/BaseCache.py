from abc import ABC, abstractmethod
from utils.logs.log_utils import debug, info


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
        # initial message
        info("🔄 Base cache initialization started...")

        # initialize data
        try:
            if cache_class is not None:
                self.cache = cache_class(
                    config_settings.cache_size,
                    callback=self._on_evict
                )
            else:
                self.cache = None
            self.maxsize = config_settings.cache_size
            self.ttl = config_settings.ttl
            self.metrics_logger = metrics_logger
            self.store = {}
            self.expiry = {}
            self.scores = {}
            self._last_put_time = None
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except ValueError as e:
            raise ValueError(f"ValueError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # print a successful message
        info("🟢 Base cache initialized.")


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
        # initial message
        info("🔄 Expiration check started...")

        try:
            # print a successful message
            info("🟢 Expiration check finished.")

            return (
                    key in self.expiry and
                    self.expiry[key] < current_time
            )
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except KeyError as e:
            raise KeyError(f"KeyError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")


    def _remove_expired_keys(
            self,
            current_time
    ):
        """
        Method to remove expired keys from the cache.
        :param current_time: The current time.
        :return:
        """
        # initial message
        info("🔄 Expired key removal started...")

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
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except KeyError as e:
            raise KeyError(f"KeyError: {e}.")
        except ValueError as e:
            raise ValueError(f"ValueError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # print a successful message
        info("🟢 Expired key removed.")


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
        # initial message
        info("🔄 Key access started...")

        try:
            # trace event
            self.metrics_logger.log_get(
                key,
                current_time
            )

            # check if the key is in the cache
            if (
                self.cache is not None and
                key in self.cache and not
                self._is_expired(
                    key,
                    current_time
                )
            ):
                _ = self.cache[key]

                # debugging
                debug(f"⚙️Key: {key} contained in cache.")
                # print a successful message
                info("🟢 Key access finished.")

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
                # print a successful message
                info("🟢 Key access finished.")

                return True
            else:
                # print a successful message
                info("🟢 Key access finished.")

                return False
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except KeyError as e:
            raise KeyError(f"KeyError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")


    def _on_evict(
            self,
            key
    ):
        """
        Callback triggered by cachetools when a key is evicted.
        :param key: The key to evict.
        :return:
        """
        try:
            # evict key
            self.expiry.pop(key, None)

            # trace event
            self.metrics_logger.log_eviction(
                key,
                self._last_put_time
            )
            debug(f"⚙️ Key {key} evicted by cachetools.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}")


    @abstractmethod
    def put(self, *args, **kwargs):
        """
        Each strategy defines its own put method.
        :param self:
        :param args:
        :param kwargs:
        :return:
        """