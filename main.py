import logging
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from validation import validation

logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s"
)

data_generation()

data_preprocessing()

validation()