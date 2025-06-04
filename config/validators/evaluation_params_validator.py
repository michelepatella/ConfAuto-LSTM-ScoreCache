from config.config_io.config_reader import get_config
from utils.logs.log_utils import info


def _check_evaluation_params(top_k):
    """
    Method to check evaluation parameters.
    :param top_k: The top-k value for computing top-k accuracy.
    :return:
    """
    # check top-k
    if (
        not isinstance(top_k, int) or
        top_k <= 0
    ):
        raise RuntimeError("❌ 'evaluation.top_k' must be an integer > 0.")


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
    _check_evaluation_params(
        top_k
    )

    # show a successful message
    info("🟢 Evaluation general params validated.")

    return top_k