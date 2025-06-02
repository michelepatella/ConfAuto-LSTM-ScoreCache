import random
from simulation.caches.BaseCache import BaseCache
from utils.log_utils import debug


class RandomCache(BaseCache):

    def put(
            self,
            key,
            ttl,
            current_time
    ):
        """
        Method to put a key in the random cache.
        :param key: The key to put.
        :param ttl: The TTL of the key.
        :param current_time: The current time.
        :return:
        """
        try:
            # clean up the cache removing expired keys
            self._remove_expired_keys(
                current_time
            )

            # check if the key is in cache
            if self.contains(
                    key,
                    current_time
            ):
                # update TTL of the key
                self.expiry[key] = current_time + ttl

                # trace event
                self.metrics_logger.log_put(
                    key,
                    current_time,
                    ttl
                )

                # debugging
                debug(f"⚙️ Key {key} already cached, new TTL: {self.expiry[key]}.")
                return

            # check if the cache is full
            elif len(self.store) >= self.maxsize:

                # evict a key randomly
                evict_key = random.choice(
                    list(self.store.keys())
                )
                self.store.pop(evict_key)
                self.expiry.pop(evict_key)

                # debugging
                debug(f"⚙️Full cache, evicting: {evict_key}.")

                # trace event
                self.metrics_logger.log_eviction(
                    evict_key,
                    current_time
                )

            # store the new key
            self.store[key] = key
            self.expiry[key] = current_time + ttl

            # trace event
            self.metrics_logger.log_put(
                key,
                current_time,
                ttl
            )

            # debugging
            debug(f"⚙️Key {key} cached with TTL: {self.expiry[key]}.")

        except (
                AttributeError,
                TypeError,
                KeyError,
                IndexError,
                ValueError
        ) as e:
            raise RuntimeError(f"❌ Error while caching the key: {e}.")