def check_evaluation_params(top_k):
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
        raise RuntimeError("'evaluation.top_k' must be an integer > 0.")