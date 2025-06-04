def check_testing_params(testing_batch_size):
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