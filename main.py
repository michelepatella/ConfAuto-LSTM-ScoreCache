import logging
from data_generation import generate_zipf_dataset
from validation import validation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

generate_zipf_dataset("static")

validation()