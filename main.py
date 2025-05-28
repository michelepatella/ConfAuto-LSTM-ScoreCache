from config import prepare_config
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from simulation import simulate
from simulation.CacheWrapper import CacheWrapper
from testing import testing
from training import training
from validation import validation
from cachetools import LRUCache, LFUCache, FIFOCache
from simulation.LSTMCache import LSTMCache
from simulation.RandomCache import RandomCache


config_settings = prepare_config()

#data_generation(config_settings)

#data_preprocessing(config_settings)

#config_settings = validation(config_settings)

#training(config_settings)

#avg_loss, metrics = testing(config_settings)
"""
print("\n" + "="*85)
print(" " * 30 + "Model Standalone Evaluation Report")
print("="*85 + "\n")
print(f"Average Loss:       {avg_loss:.4f}\n")
print("📉 Class Report per Class:")
print(metrics['class_report'] + "\n")
print(f"Top-k Accuracy:     {metrics['top_k_accuracy']:.4f}")
print(f"Kappa Statistic:    {metrics['kappa_statistic']:.4f}")
print("\n" + "="*85 + "\n")
"""

# setup cache strategies
strategies = {
    'LRU': CacheWrapper(LRUCache, config_settings),
    'LFU': CacheWrapper(LFUCache, config_settings),
    'FIFO': CacheWrapper(FIFOCache, config_settings),
    'RANDOM': RandomCache(config_settings),
    'LSTM': LSTMCache(config_settings),
}

# run simulation
results = []
for policy, cache in strategies.items():
    result = simulate(cache, policy, config_settings)
    results.append(result)

# show results
print("\n" + "="*90)
print(" " * 30 + "Overall System Evaluation Report")
print("="*90 + "\n")
print(f"{'Policy':<25} | {'Hit Rate (%)':>12} | {'Miss Rate (%)':>13}")
print("-"*90)
for res in results:
    print(f"{res['policy']:<25} | {res['hit_rate']:>12.2f} | {res['miss_rate']:>13.2f}")
print("\n" + "="*90 + "\n")