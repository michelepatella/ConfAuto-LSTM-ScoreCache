from config.config_io.config_reader import get_config
from config.validation.model_params.model_params_checker import check_general_model_params, check_model_params
from utils.logs.log_utils import info


def validate_model_general_params(config):
    """
    Method to validate general model parameters.
    :param config: The config object.
    :return: All the model general parameters.
    """
    # initial message
    info("🔄 Model general params validation started...")

    # general
    num_features = get_config(
        config,
        "model.general.num_features"
    )
    model_save_path = get_config(
        config,
        "model.general.save_path"
    )

    # check general params
    check_general_model_params(
        num_features,
        model_save_path
    )

    # show a successful message
    info("🟢 Model general params validated.")

    return num_features, model_save_path


def validate_model_params(config):
    """
    Method to validate model parameters.
    :param config: The config object.
    :return: All the model parameters.
    """
    # initial message
    info("🔄 Model params validation started...")

    # params
    model_params = get_config(
        config,
        "model.params"
    )
    hidden_size = get_config(
        config,
        "model.params.hidden_size"
    )
    num_layers = get_config(
        config,
        "model.params.num_layers"
    )
    bias = get_config(
        config,
        "model.params.bias"
    )
    batch_first = get_config(
        config,
        "model.params.batch_first"
    )
    dropout = get_config(
        config,
        "model.params.dropout"
    )
    bidirectional = get_config(
        config,
        "model.params.bidirectional"
    )
    proj_size = get_config(
        config,
        "model.params.proj_size"
    )

    # check model params
    check_model_params(
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        proj_size
    )

    # show a successful message
    info("🟢 Model params validated.")

    return (
        model_params,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        proj_size
    )