from utils.config_utils import get_config_value
from utils.log_utils import info


def _check_evaluation_params(top_k, fp_cost, fn_cost):
    """
    Method to check evaluation parameters.
    :param top_k: The top-k value for computing top-k accuracy.
    :param fp_cost: The false positive cost.
    :param fn_cost: The false negative cost.
    :return:
    """
    # check top-k
    if (
        not isinstance(top_k, int) or
        top_k <= 0
    ):
        raise RuntimeError("❌ 'evaluation.top_k' must be an integer > 0.")

    # check costs (both false positive and false negative ones)
    if not isinstance(fp_cost, float) or fp_cost < 0:
        raise RuntimeError("❌ 'evaluation.costs.fp_cost' must be a float >= 0.")
    if not isinstance(fn_cost, float) or fn_cost < 0:
        raise RuntimeError("❌ 'evaluation.costs.fn_cost' must be a float >= 0.")


def _validate_evaluation_general_params(config):
    """
    Method to validate evaluation general parameters.
    :param config:
    :return: The evaluation general parameters.
    """
    # initial message
    info("🔄 Evaluation general params validation started...")

    top_k = get_config_value(
        config,
        "evaluation.top_k"
    )

    fp_cost = get_config_value(
        config,
        "evaluation.costs.fp_cost"
    )

    fn_cost = get_config_value(
        config,
        "evaluation.costs.fn_cost"
    )

    # check evaluation params
    _check_evaluation_params(
        top_k,
        fp_cost,
        fn_cost
    )

    # show a successful message
    info("🟢 Evaluation general params validated.")

    return top_k, fp_cost, fn_cost