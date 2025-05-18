from utils.config_utils import _get_config_value
from utils.log_utils import _info


def _check_general_model_params(
        num_features,
        model_save_path
):
    """
    Method to check general model parameters.
    :param num_features: The number of features.
    :param model_save_path: The path to save the model.
    :return:
    """
    # check number of features
    if num_features <= 0 or not isinstance(num_features, int):
        raise RuntimeError("❌ 'model.general.num_features'"
                           " must be an integer > 0.")

    # check model save path
    if not isinstance(model_save_path, str):
        raise RuntimeError("❌ 'model.general.save_path' must be a string.")


def _check_model_params(
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        proj_size
):
    # check hidden size
    if (
        not isinstance(hidden_size, int)
        or hidden_size <= 0
    ):
        raise RuntimeError("❌ 'model.params.hidden_size' must be an integer > 0.")

    # check number of layers
    if (
        not isinstance(num_layers, int)
        or num_layers <= 0
    ):
        raise RuntimeError("❌ 'model.params.num_layers' must be an integer > 0.")

    # check bias
    if not isinstance(bias, bool):
        raise RuntimeError("❌ 'model.params.bias' must be a boolean.")

    # check batch first
    if not isinstance(batch_first, bool):
        raise RuntimeError("❌ 'model.params.batch_first' must be a boolean.")

    # check dropout
    if (
        not isinstance(dropout, float)
        or not (0.0 <= dropout < 1.0)
    ):
        raise RuntimeError("❌ 'model.params.dropout' must be a "
                           "float within [0.0, 1.0).")
    if num_layers == 1 and dropout > 0:
        _info("ℹ️ 'dropout' is ignored when 'num_layers' == 1.")

    # check bidirectional
    if not isinstance(bidirectional, bool):
        raise RuntimeError("❌ 'model.params.bidirectional' must be a boolean.")

    # check project size
    if (
        not isinstance(proj_size, int)
        or proj_size < 0 or
        proj_size > hidden_size
    ):
        raise RuntimeError("❌ 'model.params.proj_size' must be an"
                           " integer in [0, hidden_size].")


def _validate_model_general_params(config):
    """
    Method to validate general model parameters.
    :param config: The config object.
    :return: All the model general parameters.
    """
    # initial message
    _info("🔄 Model general params validation started...")

    # general
    num_features = _get_config_value(
        config,
        "model.general.num_features"
    )
    model_save_path = _get_config_value(
        config,
        "model.general.save_path"
    )

    # check general params
    _check_general_model_params(
        num_features,
        model_save_path
    )

    # show a successful message
    _info("🟢 Model general params validated.")

    return num_features, model_save_path


def _validate_model_params(config):
    """
    Method to validate model parameters.
    :param config: The config object.
    :return: All the model parameters.
    """
    # initial message
    _info("🔄 Model params validation started...")

    # params
    model_params = _get_config_value(
        config,
        "model.params"
    )
    hidden_size = _get_config_value(
        config,
        "model.params.hidden_size"
    )
    num_layers = _get_config_value(
        config,
        "model.params.num_layers"
    )
    bias = _get_config_value(
        config,
        "model.params.bias"
    )
    batch_first = _get_config_value(
        config,
        "model.params.batch_first"
    )
    dropout = _get_config_value(
        config,
        "model.params.dropout"
    )
    bidirectional = _get_config_value(
        config,
        "model.params.bidirectional"
    )
    proj_size = _get_config_value(
        config,
        "model.params.proj_size"
    )

    # check model params
    _check_model_params(
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        proj_size
    )

    # show a successful message
    _info("🟢 Model params validated.")

    return (model_params, hidden_size, num_layers, bias,
            batch_first, dropout, bidirectional, proj_size)