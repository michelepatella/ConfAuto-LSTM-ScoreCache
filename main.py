import logging
from data_generation import generate_zipf_dataset
from preprocessing.main import preprocessing
from validation import validation

dataset_type = "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

generate_zipf_dataset(dataset_type)

preprocessing(dataset_type)

validation()