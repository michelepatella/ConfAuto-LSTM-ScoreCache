from utils.config_utils import _get_config_value


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


def _validate_evaluation_general_params():
    """
    Method to validate evaluation general parameters.
    :return: The evaluation general parameters.
    """
    top_k = _get_config_value("evaluation.top_k")

    # check evaluation params
    _check_evaluation_params(top_k)

    return top_k