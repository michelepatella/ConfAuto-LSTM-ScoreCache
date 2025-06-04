from config.config_io.config_reader import get_config
from config.validators.training_params.training_params_checker import check_general_training_params, \
    check_optimizer_params, check_training_early_stopping_params
from utils.logs.log_utils import info


def validate_training_general_params(config):
    """
    Method to validate general training parameters.
    :param config: The config object.
    :return: All the general training parameters.
    """
    # initial message
    info("🔄 Training general params validation started...")

    # general
    training_num_epochs = get_config(
        config,
        "training.general.num_epochs"
    )
    training_batch_size = get_config(
        config,
        "training.general.batch_size"
    )

    # check general training params
    check_general_training_params(
        training_num_epochs,
        training_batch_size
    )

    # show a successful message
    info("🟢 Training general params validated.")

    return (
        training_num_epochs,
        training_batch_size
    )


def validate_training_optimizer_params(config):
    """
    Method to validate training optimizer parameters.
    :param config: The config object.
    :return: All the training optimizer parameters.
    """
    # initial message
    info("🔄 Training optimizer params validation started...")

    # optimizer
    optimizer_type = get_config(
        config,
        "training.optimizer.type"
    )
    learning_rate = get_config(
        config,
        "training.optimizer.learning_rate"
    )
    weight_decay = get_config(
        config,
        "training.optimizer.weight_decay"
    )
    momentum = get_config(
        config,
        "training.optimizer.momentum"
    )

    # check optimizer params
    check_optimizer_params(
        optimizer_type,
        learning_rate,
        weight_decay,
        momentum
    )

    # show a successful message
    info("🟢 Training optimizer params validated.")

    return (
        optimizer_type,
        learning_rate,
        weight_decay,
        momentum
    )


def validate_training_early_stopping_params(config):
    """
    Method to validate training early stopping parameters.
    :param config: Config object.
    :return: All early stopping parameters.
    """
    # initial message
    info("🔄 Training early stopping params validation started...")

    # early stopping
    training_early_stopping_patience = get_config(
        config,
        "training.early_stopping.patience"
    )
    training_early_stopping_delta = get_config(
        config,
        "training.early_stopping.delta"
    )

    # check early stopping params
    check_training_early_stopping_params(
        training_early_stopping_patience,
        training_early_stopping_delta
    )

    # show a successful message
    info("🟢 Training early stopping params validated.")

    return (
        training_early_stopping_patience,
        training_early_stopping_delta
    )