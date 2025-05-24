sim_time = 0.0

for request in test_data:
    delta = request["delta_t"]
    sim_time += delta

    key = str(request["key"])
    features = request["features"]

    # Intelligent Cache
    intelligent_cache.serve_request(key, features)

    # Baseline Cache
    baseline_cache.serve_request(key, features)

    # Simula tempo reale per Redis
    time.sleep(0.01)  # Redis usa secondi reali, quindi fai attenzione