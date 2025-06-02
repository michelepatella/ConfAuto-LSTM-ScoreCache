from simulation.caches.BaseCache import BaseCache
from utils.log_utils import debug, info


class CacheWrapper(BaseCache):

    def put(
            self,
            key,
            ttl,
            current_time
    ):
        """
        Method to put a key into the cache.
        :param key: The key to put.
        :param ttl: The TTL for the key.
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

            # put the key in  cache
            self.cache[key] = key

            # set TTL for the key
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
                TypeError
        ) as e:
            raise RuntimeError(f"❌ Error while caching the key: {e}.")

        # print a successful message
        info("🟢 Key inserted.")