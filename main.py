import numpy as np
from config import prepare_config
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from testing import testing
from training import training
from validation import validation


config_settings = prepare_config()

#data_generation(config_settings)

#data_preprocessing(config_settings)

#config_settings = validation(config_settings)

#training(config_settings)

avg_loss, avg_loss_per_class, metrics, cost_perc = testing(config_settings)

print("----------------------------------------------------------------------------------------")
print(f"Average Loss: {avg_loss}")
print(f"Average Loss per Class:")
print(f"{avg_loss_per_class}")

print(f"📉 Class Report per Class:")
print(f"{metrics['class_report']}")

print(f"\nConfusion Matrix:\n{np.array(metrics['confusion_matrix'])}")

print(f"Top-k Accuracy: {metrics['top_k_accuracy']}")
print(f"Kappa Statistic: {metrics['kappa_statistic']}")

print(f"📉 Cost (%): {cost_perc}")