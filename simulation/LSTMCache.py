class LSTMCache:

    def __init__(self, config_settings):
        """
        Method to initialize the LSTM cache.
        :param config_settings: The configuration settings.
        :return:
        """
        self.maxsize = config_settings.cache_size
        self.store = {}
        self.expiry = {}
        self.threshold_prob = config_settings.threshold_prob
        self.confidence_threshold = config_settings.threshold_ci
        self.ttl_base = config_settings.ttl_base
        self.alpha = config_settings.alpha_ttl
        self.beta = config_settings.beta_ttl
        self.probs = {}
        self.confidences = {}


    def _insert_or_update_entry(
            self,
            key,
            prob,
            confidence,
            current_time
    ):
        """
        Method to manage inserting or updating an entry, containing the full
        logic of prefetching, TTL assigment, and eviction.
        :param key: The key.
        :param prob: The probability of the key of being accessed.
        :param confidence: The confidence related to the prediction.
        :param current_time: The current time.
        :return:
        """
        # check if the key must be removed since has lower
        # probability or unreliable prediction
        if (
            prob < self.threshold_prob or
            confidence < self.confidence_threshold
        ):
            # remove the key from the cache
            self.store.pop(key, None)
            self.expiry.pop(key, None)
            self.probs.pop(key, None)
            self.confidences.pop(key, None)
            return

        # compute TTL dynamically
        ttl = self.ttl_base * (1 + self.alpha * prob) * (1 + self.beta * confidence)
        if ttl <= 0:
            ttl = self.ttl_base

        # if the key is new and the cache is full, evict something
        if (
            key not in self.store and
            len(self.store) >= self.maxsize
        ):
            # identify key candidates to be evicted
            evict_candidates = [
                k for k in self.store.keys()
                if self.probs.get(k, 0) < self.threshold_prob or
                   self.confidences.get(k, 0) < self.confidence_threshold
            ]

            # check if any key candidates to be evicted
            if not evict_candidates:
                # if no candidate, evict the least confident key
                evict_keys = [min(
                    self.store.keys(),
                    key=lambda k: self.probs.get(k, 0) * self.confidences.get(k, 0)
                )]
            else:
                evict_keys = evict_candidates

            # evict all candidate keys
            for k in evict_keys:
                self.store.pop(k, None)
                self.expiry.pop(k, None)
                self.probs.pop(k, None)
                self.confidences.pop(k, None)

        # put/update the entry
        self.store[key] = key
        self.expiry[key] = current_time + ttl
        self.probs[key] = prob
        self.confidences[key] = confidence


    def contains(self, key, current_time):
        """
        Method to check if a key is in the cache.
        :param key: The key to check.
        :param current_time: The current time.
        :return: True if the key is in the cache, False otherwise.
        """
        return (
                key in self.store and
                self.expiry.get(key, 0) > current_time
        )


    def get(self, key, current_time):
        """
        Method to get the key from the cache.
        :param key: The key to get.
        :param current_time: The current time.
        :return: The key from the cache if present, None otherwise.
        """
        # check if the key is in the cache
        if self.contains(key, current_time):
            return self.store[key]
        return None


    def put(
            self,
            key,
            prob,
            confidence,
            current_time
    ):
        """
        Method to orchestrate putting a key into the cache.
        :param key: The key to put.
        :param prob: The probability of the key of being accessed.
        :param confidence: The confidence related to the prediction.
        :param current_time: The current time.
        :return:
        """
        self._insert_or_update_entry(
            key,
            prob,
            confidence,
            current_time
        )


    def update_prediction_for_key(
            self,
            key,
            prob,
            confidence,
            current_time
    ):
        """
        Method to update the prediction for a key.
        :param key: The key to update.
        :param prob: The probability of the key of being accessed.
        :param confidence: The confidence related to the prediction.
        :param current_time: The current time.
        :return:
        """
        if key in self.store:
            self._insert_or_update_entry(
                key,
                prob,
                confidence,
                current_time
            )