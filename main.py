import logging
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from testing import testing
from training import training
from utils.log_utils import phase_var
from validation import validation


data_generation()

data_preprocessing()

validation()

training()

avg_loss, avg_loss_per_class, metrics = testing()

print("----------------------------------------------------------------------------------------")
print(f"Average loss: {avg_loss}")
print(f"Average loss per class: {avg_loss_per_class}")
print(f"Metrics: {metrics}")