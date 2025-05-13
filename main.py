import logging
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from training_with_validation import training_with_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

data_generation()

data_preprocessing()

training_with_validation()