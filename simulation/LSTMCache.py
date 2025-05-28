class LSTMCache:

    def __init__(
            self,
            maxsize,
            threshold_prob,
            confidence_threshold,
            ttl_base,
            alpha,
            beta
    ):
        """
        Method to initialize the LSTM cache.
        :param maxsize: The maximum size of the cache.
        :param threshold_prob: The threshold probability of the LSTM cache.
        :param confidence_threshold: The confidence intervals threshold
        of the LSTM cache.
        :param ttl_base: The TTL base of the LSTM cache (to calculate TTL).
        :param alpha: The alpha parameter of the LSTM cache (to calculate TTL).
        :param beta: The beta parameter of the LSTM cache (to calculate TTL).
        """
        self.maxsize = maxsize
        self.store = {}
        self.expiry = {}
        self.threshold_prob = threshold_prob
        self.confidence_threshold = confidence_threshold
        self.ttl_base = ttl_base
        self.alpha = alpha
        self.beta = beta
        self.probs = {}
        self.confidences = {}


    def update_predictions(self, predictions):
        """
        Update the stored probabilities and confidence intervals
        for all keys already in the cache.
        :param predictions: dict in the form {key: (prob, conf_int)}
        :return:
        """
        # update probabilities and CIs
        for key, (prob, conf_int) in predictions.items():
            if key in self.store:
                self.probs[key] = prob
                self.confidences[key] = conf_int


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


    def put(self, key, prob, conf_int, current_time):
        """
        Method to put the key in the cache and update this latter.
        :param key: The key to put.
        :param prob: The probability of the key returned by the LSTM model.
        :param conf_int: The confidence interval related to the key predicted
        by the LSTM model.
        :param current_time: The current time.
        :return:
        """
        # check if the key is already in the cache
        if key in self.store:
            # evict the key in case probability less than
            # a predefined threshold or CIs less
            # than a predefined threshold
            if (
                prob < self.threshold_prob or
                conf_int < self.confidence_threshold
            ):
                self.store.pop(key)
                self.expiry.pop(key, None)
                self.probs.pop(key, None)
                self.confidences.pop(key, None)
                return

        # prefetch/update in case probability greater than
        # a predefined threshold and CIs greater than a
        # a predefined threshold
        if (
            prob > self.threshold_prob and
            conf_int > self.confidence_threshold
        ):
            # calculate a dynamic TTL for the key to
            # be inserted
            ttl = (
                    self.ttl_base * (1 + self.alpha * prob)
                   * (1 + self.beta * conf_int)
            )
            if ttl < 0:
                ttl = 0

            # if the cache is full, evict keys whose
            # probability is decreased below the predefined
            # threshold or whose CIs are decreased below the predefined
            # threshold
            if (
                len(self.store) >= self.maxsize
                and key not in self.store
            ):
                # identify keys to evict
                evict_candidates = [
                    k for k in self.store.keys()
                    if self.probs.get(k, 0) < self.threshold_prob or
                       self.confidences.get(k, 0) < self.confidence_threshold
                ]

                # if there is at least one evict candidate
                if evict_candidates:
                    evict_key = evict_candidates[0]
                else:
                    # if there is not any evict candidate
                    # identify the one having the lowest prob*conf
                    evict_key = min(
                        self.store.keys(),
                        key=lambda k: self.probs.get(k, 0) *
                                      self.confidences.get(k, 0)
                    )

                # remove the key(s)
                self.store.pop(evict_key)
                self.expiry.pop(evict_key, None)
                self.probs.pop(evict_key, None)
                self.confidences.pop(evict_key, None)

            # put/update the key
            self.store[key] = key
            self.expiry[key] = current_time + ttl
            self.probs[key] = prob
            self.confidences[key] = conf_int