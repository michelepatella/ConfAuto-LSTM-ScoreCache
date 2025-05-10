from data_generation.dataset_generator import _generate_static_dataset, _generate_dynamic_dataset


def generate_zipf_dataset(distribution_type):
    """
    Method to orchestrate the (static or dynamic) zipf dataset generation.
    :param distribution_type: Zipf distribution's type (static or dynamic).
    :return:
    """
    # generate a static dataset
    if distribution_type == "static":
        _generate_static_dataset()

    # generate a dynamic dataset
    elif distribution_type == "dynamic":
        _generate_dynamic_dataset()

    # handle errors
    else:
        raise ValueError("Unknown distribution type.")