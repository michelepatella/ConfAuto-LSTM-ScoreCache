from simulation.caches.BaseCache import BaseCache
from utils.logs.log_utils import debug, info


class CacheWrapper(BaseCache):

    def put(
            self,
            key,
            current_time
    ):
        """
        Method to put a key into the cache.
        :param key: The key to put.
        :param current_time: The current time.
        :return:
        """
        # initial message
        info("🔄 Key insertion started...")

        try:
            # update the current time
            self._last_put_time = current_time

            # clean up the cache removing expired keys
            self._remove_expired_keys(
                current_time
            )

            # put the key in cache
            self.cache[key] = key

            # set expiration time for the key
            self.expiry[key] = current_time + self.ttl

            # trace event
            self.metrics_logger.log_put(
                key,
                current_time,
                self.ttl
            )

            # debugging
            debug(f"⚙️Key {key} cached.")

        except (
                AttributeError,
                TypeError
        ) as e:
            raise RuntimeError(f"❌ Error while caching the key: {e}.")

        # print a successful message
        info("🟢 Key inserted.")