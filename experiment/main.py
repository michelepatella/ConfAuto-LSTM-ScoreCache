from cachetools import LRUCache, LFUCache, FIFOCache
import pandas as pd
import math

from experiment.LSTMCache import LSTMCache
from experiment.RandomCache import RandomCache


CACHE_SIZE = 10
TTL = 120

CSV_FILE = '/Users/michelepatella/Personal_Projects/CLIC/data/static_access_logs.csv'
df = pd.read_csv(CSV_FILE)

def simulate(cache, policy_name):
    # initialize data
    hits = 0
    misses = 0
    time_map = {}
    access_counter = 0
    PREDICTION_INTERVAL = 20
    all_keys_seen = set()

    # for each request
    for _, row in df.iterrows():

        # get the requested key
        key = row['request']

        # calculate timestamps in seconds
        angle = math.atan2(row['sin_time'], row['cos_time'])
        if angle < 0:
            angle += 2 * math.pi
        current_time = angle / (2 * math.pi) * 24 * 3600

        # if the LSTM cache is being used
        if policy_name == 'LSTM':
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

        # if the random cache is being used
        if policy_name == 'RANDOM':
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
        else:
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

    # calculate hit rate and miss rate in terms of %
    total = hits + misses
    hit_rate = hits / total * 100
    miss_rate = misses / total * 100

    return {
        'policy': policy_name,
        'hit_rate': hit_rate,
        'miss_rate': miss_rate,
        'hits': hits,
        'misses': misses
    }

# setup cache strategies
strategies = {
    'LRU': LRUCache(maxsize=CACHE_SIZE),
    'LFU': LFUCache(maxsize=CACHE_SIZE),
    'FIFO': FIFOCache(maxsize=CACHE_SIZE),
    'RANDOM': RandomCache(maxsize=CACHE_SIZE),
    'LSTM': LSTMCache(
        maxsize=CACHE_SIZE,
        threshold_prob=0.6,
        confidence_threshold=0.6,
        ttl_base=60,
        alpha=1.0,
        beta=1.0
    ),
}

# run simulation
results = []
for policy, cache in strategies.items():
    result = simulate(cache, policy)
    results.append(result)

# visualize results
print("=== Risultati ===")
for res in results:
    print(f"{res['policy']}: Hit Rate = {res['hit_rate']:.2f}%, Miss Rate = {res['miss_rate']:.2f}%")