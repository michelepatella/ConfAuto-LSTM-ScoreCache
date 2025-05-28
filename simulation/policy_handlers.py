def handle_lstm_policy(
        cache,
        key,
        current_time,
        access_counter,
        PREDICTION_INTERVAL,
        hits,
        misses
):
    access_counter += 1

    # check if the cache contains the key
    if cache.contains(key, current_time):
        # increment cache hits
        hits += 1
        print(f"Time: {current_time:.2f} | Key: {key} | HIT")
    else:
        # increment cache misses
        misses += 1
        print(f"Time: {current_time:.2f} | Key: {key} | MISS")

        # make a prediction using LSTM model
        predictions = mock_lstm_model_predict([key])
        prob, conf_int = predictions[key]
        cache.put(key, prob, conf_int, current_time)

    # if the access counter exceeds the
    # interval, make a prediction and update probs
    # and CIs of predictions
    if access_counter >= PREDICTION_INTERVAL:
        batch_preds = mock_lstm_model_predict(all_keys_seen)
        cache.update_predictions(batch_preds)
        access_counter = 0


def handle_random_policy(
        cache,
        key,
        current_time,
        TTL,
        hits,
        misses
):
    # check if the key is in the cache
    if cache.contains(key, current_time):
        # increment cache hits
        hits += 1
        print(f"Time: {current_time:.2f} | Key: {key} | HIT")
    else:
        # increment cache misses
        misses += 1
        print(f"Time: {current_time:.2f} | Key: {key} | MISS")
    # put the key in the cache
    cache.put(key, TTL, current_time)


def handle_default_policy(
        cache,
        key,
        current_time,
        TTL,
        time_map,
        hits,
        misses
):
    # check if the key is in the cache and
    # its TTL is not expired
    if key in cache and time_map.get(key, 0) > current_time:
        # increment cache hits
        hits += 1
        print(f"Time: {current_time:.2f} | Key: {key} | HIT")
    else:
        # increment cache misses, set the key into the cache,
        # and assign it a TTL
        misses += 1
        cache[key] = key
        time_map[key] = current_time + TTL

        print(f"Time: {current_time:.2f} | Key: {key} | MISS")