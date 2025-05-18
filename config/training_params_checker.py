def _check_general_training_params(
        training_num_epochs,
        training_batch_size
):
    """
    Method to check the general training parameters.
    :param training_num_epochs: The number of training epochs.
    :param training_batch_size: The size of the training batch.
    :return:
    """
    # check number of epochs
    if (
        not isinstance(training_num_epochs, int)
        or training_num_epochs <= 0
    ):
        raise RuntimeError("❌ 'training.general.num_epochs' "
                           "must be an integer > 0.")

    # check training batch size
    if (
        not isinstance(training_batch_size, int)
        or training_batch_size <= 0
    ):
        raise RuntimeError("❌ 'training.general.batch_size' "
                           "must be an integer > 0.")


def _check_optimizer_params(
        optimizer_type,
        learning_rate,
        weight_decay,
        momentum
):
    """
    Method to check the optimizer parameters.
    :param optimizer_type: The type of optimizer.
    :param learning_rate: The learning rate value.
    :param weight_decay: The weight decay value.
    :param momentum: The momentum value.
    :return:
    """
    # check optimizer type
    if optimizer_type not in {"adam", "adamw", "sgd"}:
        raise RuntimeError("❌ 'training.optimizer.type' "
                           "must 'adam', 'adamw', or 'sgd'.")

    # check learning rate
    if (
        not isinstance(learning_rate, float)
        or learning_rate <= 0
    ):
        raise RuntimeError("❌ 'training.optimizer.learning_rate'"
                           " must be a float > 0.")

    # check weight decay
    if (
        not isinstance(weight_decay, float)
        or weight_decay < 0
    ):
        raise RuntimeError("❌ 'training.optimizer.weight_decay' "
                           "must be a float >= 0.")

    # check momentum
    if (
        not isinstance(momentum, float)
        or not (0 <= momentum <= 1)
    ):
        raise RuntimeError("❌ 'training.optimizer.momentum' must be"
                           " a float between 0.0 and 1.0.")