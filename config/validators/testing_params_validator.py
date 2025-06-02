from utils.config_utils import get_config_value
from utils.log_utils import info


def _check_testing_params(testing_batch_size):
    """
    Method to check testing parameters.
    :param testing_batch_size: The testing batch size.
    :return:
    """
    # check batch size
    if (
        not isinstance(testing_batch_size, int)
        or testing_batch_size <= 0
    ):
        raise RuntimeError("❌ 'testing.general.batch_size' must be an integer > 0.")


def _validate_testing_general_params(config):
    """
    Method to validate testing general parameters.
    :param config: The config object.
    :return: All the testing general parameters.
    """
    # initial message
    info("🔄 Testing general params validation started...")

    testing_batch_size = get_config_value(
        config,
        "testing.general.batch_size"
    )

    # check testing params
    _check_testing_params(testing_batch_size)

    # show a successful message
    info("🟢 Testing general params validated.")

    return testing_batch_size