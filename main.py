import logging
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from testing import testing
from training import training
from validation import validation

"""
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s"
)
"""

data_generation()

data_preprocessing()

# validation()

training()

avg_loss, metrics = testing()

print("----------------------------------------------------------------------------------------")
print(f"Average loss: {avg_loss}")
print(f"Metrics: {metrics}")