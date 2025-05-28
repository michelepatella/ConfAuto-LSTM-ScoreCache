from config import prepare_config
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from simulation import simulate
from testing import testing
from training import training
from validation import validation
from cachetools import LRUCache, LFUCache, FIFOCache
from simulation.LSTMCache import LSTMCache
from simulation.RandomCache import RandomCache


config_settings = prepare_config()

#data_generation(config_settings)

#data_preprocessing(config_settings)

config_settings = validation(config_settings)

#training(config_settings)

#avg_loss, metrics = testing(config_settings)
""""
print("----------------------------- Model Standalone Evaluation -----------------------------")

print(f"Average Loss: {avg_loss}")

print(f"📉 Class Report per Class:")
print(f"{metrics['class_report']}")

print(f"Top-k Accuracy: {metrics['top_k_accuracy']}")
print(f"Kappa Statistic: {metrics['kappa_statistic']}")

print("---------------------------------------------------------------------------------------")
"""
""""
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
print("------------------------------ Overall System Evaluation ------------------------------")
for res in results:
    print(f"{res['policy']}: Hit Rate = {res['hit_rate']:.2f}%, Miss Rate = {res['miss_rate']:.2f}%")
print("---------------------------------------------------------------------------------------")
"""