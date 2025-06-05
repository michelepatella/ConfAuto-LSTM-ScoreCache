from config.config_io.config_reader import get_config
from config.validation.inference_params.inference_params_checker import check_confidence_intervals_params
from utils.logs.log_utils import info


def validate_inference_confidence_intervals_params(config):
    """
    Method to validate confidence intervals params
    :param config: The config object.
    :return: All the confidence intervals params.
    """
    # initial message
    info("🔄 Confidence intervals params validation started...")

    # params
    confidence_level = get_config(
        config,
        "inference.confidence_intervals.confidence_level",
    )
    mc_dropout_num_samples = get_config(
        config,
        "inference.mc_dropout.num_samples",
    )

    # check confidence intervals params
    check_confidence_intervals_params(
        confidence_level,
        mc_dropout_num_samples
    )

    # show a successful message
    info("🟢 Confidence intervals params validated.")

    return confidence_level, mc_dropout_num_samples