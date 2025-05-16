import logging
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from testing import testing
from training import training
from validation import validation


"""logging.basicConfig(
   level=logging.DEBUG,
   format="%(asctime)s - %(levelname)s - %(message)s"
)
"""
data_generation()

data_preprocessing()

#validation()

training()

avg_loss, avg_loss_per_class, metrics = testing()

print("----------------------------------------------------------------------------------------")
print(f"Average loss: {avg_loss}")
print(f"Average loss per class: {avg_loss_per_class}")
print(f"Metrics: {metrics}")