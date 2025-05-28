import pandas as pd


def simulation_setup():
    CACHE_SIZE = 10
    TTL = 120

    hits = 0
    misses = 0
    time_map = {}
    access_counter = 0
    PREDICTION_INTERVAL = 20
    all_keys_seen = set()

    CSV_FILE = '/Users/michelepatella/Personal_Projects/CLIC/data/static_access_logs.csv'
    df = pd.read_csv(CSV_FILE)

    return (
        CACHE_SIZE,
        TTL,
        df,
        hits,
        misses,
        time_map,
        access_counter,
        PREDICTION_INTERVAL,
        all_keys_seen
    )