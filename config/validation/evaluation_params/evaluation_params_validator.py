from config.config_io.config_reader import get_config
from config.validation.evaluation_params.evaluation_params_checker import check_evaluation_params
from utils.logs.log_utils import info


def validate_evaluation_general_params(config):
    """
    Method to validate evaluation general parameters.
    :param config:
    :return: The evaluation general parameters.
    """
    # initial message
    info("🔄 Evaluation general params validation started...")

    top_k = get_config(
        config,
        "evaluation.top_k"
    )

    # check evaluation params
    check_evaluation_params(
        top_k
    )

    # show a successful message
    info("🟢 Evaluation general params validated.")

    return top_k