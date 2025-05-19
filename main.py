from config import prepare_config
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from testing import testing
from training import training
from validation import validation


config_settings = prepare_config()

data_generation(config_settings)

data_preprocessing(config_settings)

config_settings = validation(config_settings)

training(config_settings)

avg_loss, avg_loss_per_class, metrics = testing(config_settings)

print("----------------------------------------------------------------------------------------")
print(f"Average loss: {avg_loss}")
print(f"Average loss per class: {avg_loss_per_class}")
print(f"Metrics: {metrics}")