import random
from simulation.caches.utils.BaseCache import BaseCache
from utils.logs.log_utils import debug, info


class LSTMCache(BaseCache):

    def __init__(
            self,
            cache_class,
            metrics_logger,
            config_settings
    ):
        """
        Method to initialize the LSTM cache.
        :param config_settings: The configuration settings.
        :param metrics_logger: The metrics logger.
        :return:
        """
        # initial message
        info("🔄 LSTM-based cache initialization started...")

        # initialize data
        super().__init__(
            cache_class,
            metrics_logger,
            config_settings
        )
        try:
            self.threshold_score = config_settings.threshold_score
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # debugging
        debug(f"⚙️Threshold score: {self.threshold_score}.")

        # print a successful message
        info("🟢 LSTM-based cache initialized.")


    def _handle_cold_start(
            self,
            key,
            score,
            current_time
    ):
        """
        Method to put a key in cache when it's cold start.
        :param key: The key to put in cache.
        :param score: The score of the key.
        :param current_time: The current time.
        :return:
        """
        # initial message
        info("🔄 Cold start management started...")

        # check if the key is in cache
        if self.contains(
                key,
                current_time
        ):
            # update expiration time of the key
            self.expiry[key] = current_time + self.ttl

            # trace event
            self.metrics_logger.log_put(
                key,
                current_time,
                self.ttl
            )

            # debugging
            debug(f"⚙️ Key {key} already cached, new expiration time: {self.expiry[key]}.")
            # print a successful message
            info("🟢 Cold start managed.")
            return

        # check if the cache is full
        elif len(self.store) >= self.maxsize:

            # evict a key randomly
            evict_key = random.choice(
                list(self.store.keys())
            )

            # debugging
            debug(f"⚙️ Full cache, evicting: {evict_key}.")

            # evict the key
            self.evict_key(evict_key)

            # trace event
            self.metrics_logger.log_eviction(
                evict_key,
                current_time
            )

        # store the key
        self._put_key(
            key,
            score,
            current_time
        )

        # print a successful message
        info("🟢 Cold start managed.")


    def evict_key(self, key):
        """
        Method to evict the key from cache.
        :param key: Key to evict.
        :return:
        """
        # initial message
        info("🔄 LSTM-based cache key eviction started...")

        try:
            self.store.pop(key, None)
            self.expiry.pop(key, None)
            self.scores.pop(key, None)
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # print a successful message
        info("🟢 LSTM-based cache key evicted.")


    def _put_key(
            self,
            key,
            score,
            current_time,
    ):
        """
        Method to put a key in cache.
        :param key: The key to put in cache.
        :param score: The score of the key.
        :param current_time: The current time.
        :return:
        """
        # initial message
        info("🔄 LSTM-based cache key insertion started...")

        try:
            self.store[key] = key
            self.scores[key] = score
            self.expiry[key] = current_time + self.ttl

            self.metrics_logger.log_put(
                key,
                current_time,
                self.ttl
            )
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # print a successful message
        info("🟢 LSTM-based cache key inserted.")


    def put(
            self,
            key,
            score,
            current_time,
            cold_start=False,
            config_settings=None
    ):
        """
        Method to put a key in the LSTM-based cache. It contains
        logic for prefetching and eviction.
        :param key: The key to put.
        :param score: The score associated with the key.
        :param current_time: The current time.
        :param cold_start: Specifies whether it's cold start or not.
        :param config_settings: The configuration settings.
        :return: True if the key has been inserted, False otherwise.
        """
        # initial message
        info("🔄 Key insertion started...")

        try:
            # debugging
            debug(f"⚙️ Key: {key}, Score: {score}.")

            # clean up the cache removing expired keys
            self._remove_expired_keys(
                current_time
            )

            # if it's not cold-start
            if not cold_start:

                # evict if the key's score is less than
                # the threshold
                if (
                    score < self.threshold_score and
                    key in self.store
                ):
                    # remove the key from the cache
                    self.evict_key(key)

                    # trace event
                    self.metrics_logger.log_eviction(
                        key,
                        current_time
                    )

                    # print a successful message
                    info("🟢 Key not inserted.")

                    return False

                elif (
                    score < self.threshold_score and
                    key not in self.store
                ):
                    # print a successful message
                    info("🟢 Key not inserted.")
                    return False

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
                        # evict the key
                        self.evict_key(key_to_evict)

                        # trace event
                        self.metrics_logger.log_eviction(
                            key_to_evict,
                            current_time
                        )

                # put the key
                self._put_key(
                    key,
                    score,
                    current_time
                )

                # trace events
                self.metrics_logger.log_prefetch_prediction(
                    current_time,
                    [key]
                )

                # print a successful message
                info("🟢 Key inserted.")
                return True

            else:
                # cold-start management
                self._handle_cold_start(
                    key,
                    score,
                    current_time
                )

                # debugging
                debug(f"⚙️ Key {key} put in the cache with expiration time: {self.expiry[key]}.")

                # print a successful message
                info("🟢 Key inserted.")
                return True
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except KeyError as e:
            raise KeyError(f"KeyError: {e}.")
        except ValueError as e:
            raise ValueError(f"ValueError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")