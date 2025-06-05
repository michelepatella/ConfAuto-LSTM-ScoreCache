from config.config_io.config_reader import get_config
from config.validation.testing_params.testing_params_checker import check_testing_params
from utils.logs.log_utils import info


def validate_testing_general_params(config):
    """
    Method to validate testing general parameters.
    :param config: The config object.
    :return: All the testing general parameters.
    """
    # initial message
    info("🔄 Testing general params validation started...")

    testing_batch_size = get_config(
        config,
        "testing.general.batch_size"
    )

    # check testing params
    check_testing_params(testing_batch_size)

    # show a successful message
    info("🟢 Testing general params validated.")

    return testing_batch_size