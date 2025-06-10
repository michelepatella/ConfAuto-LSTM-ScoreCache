def check_confidence_intervals_params(
        confidence_level,
        mc_dropout_num_samples
):
    """
    Method to check the confidence interval parameters.
    :param confidence_level: Confidence level.
    :param mc_dropout_num_samples: Number of samples to use when performing MC dropout.
    :return:
    """
    # check confidence level
    if (
        not isinstance(confidence_level, float)
        or not 0.0 <= confidence_level <= 1.0
    ):
        raise RuntimeError("'inference.confidence_intervals.confidence_level' must be a float within [0.0, 1.0].")

    # check MC dropout no. of samples
    if (
        not isinstance(mc_dropout_num_samples, int)
        or not mc_dropout_num_samples > 0
    ):
        raise RuntimeError("'inference.mc_dropout.num_samples' must be an integer > 0.")