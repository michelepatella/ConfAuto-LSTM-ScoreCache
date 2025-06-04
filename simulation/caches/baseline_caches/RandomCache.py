import random
from simulation.caches.utils.BaseCache import BaseCache
from utils.logs.log_utils import debug, info


class RandomCache(BaseCache):

    def put(
            self,
            key,
            current_time
    ):
        """
        Method to put a key in the random cache.
        :param key: The key to put.
        :param current_time: The current time.
        :return:
        """
        # initial message
        info("🔄 Key insertion started...")

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
                info("🟢 Key inserted.")

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
                debug(f"⚙️ Full cache, evicting: {evict_key}.")

                # trace event
                self.metrics_logger.log_eviction(
                    evict_key,
                    current_time
                )

            # store the new key
            self.store[key] = key
            self.expiry[key] = current_time + self.ttl

            # trace event
            self.metrics_logger.log_put(
                key,
                current_time,
                self.ttl
            )

            # debugging
            debug(f"⚙️Key {key} cached with expiration time: {self.expiry[key]}.")

        except (
                AttributeError,
                TypeError,
                KeyError,
                IndexError,
                ValueError
        ) as e:
            raise RuntimeError(f"❌ Error while caching the key: {e}.")

        # print a successful message
        info("🟢 Key inserted.")