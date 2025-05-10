import logging
from data_generation.dataset_generator import _generate_static_dataset, _generate_dynamic_dataset


def generate_zipf_dataset(distribution_type):
    """
    Method to orchestrate the (static or dynamic) zipf dataset generation.
    :param distribution_type: Zipf distribution's type (static or dynamic).
    :return:
    """
    try:
        # generate a static dataset
        if distribution_type == "static":
            _generate_static_dataset()
            logging.info(f"Zipf dataset generation successfully completed.")

        # generate a dynamic dataset
        elif distribution_type == "dynamic":
            _generate_dynamic_dataset()
            logging.info(f"Zipf dataset generation successfully completed.")

        else:
            raise ValueError("Unknown distribution type.")

    except Exception as e:
        logging.error(f"Zipf dataset generation failed: {e}")