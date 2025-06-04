from utils.logs.log_utils import info


def check_general_model_params(
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
    if not (
            isinstance(num_features, int)
            and num_features > 0
    ):
        raise RuntimeError("❌ 'model.general.num_features' must be an integer > 0.")

    # check model save path
    if not isinstance(model_save_path, str):
        raise RuntimeError("❌ 'model.general.save_path' must be a string.")


def check_model_params(
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        proj_size
):
    """
    Method to check model parameters.
    :param hidden_size: The hidden size.
    :param num_layers: The number of layers.
    :param bias: The bias parameter.
    :param batch_first: Whether to use batch first or not.
    :param dropout: The dropout rate.
    :param bidirectional: Whether to apply bidirectional or not.
    :param proj_size: The projection size.
    :return:
    """
    # check integer parameters > 0
    for name, val in [
        ("hidden_size", hidden_size),
        ("num_layers", num_layers),
    ]:
        if not (
                isinstance(val, int)
                and val > 0
        ):
            raise RuntimeError(f"❌ 'model.params.{name}' must be an integer > 0.")

    # check boolean parameters
    for name, val in [
        ("bias", bias),
        ("batch_first", batch_first),
        ("bidirectional", bidirectional),
    ]:
        if not isinstance(val, bool):
            raise RuntimeError(f"❌ 'model.params.{name}' must be a boolean.")

    # check dropout float in [0.0, 1.0)
    if not (
            isinstance(dropout, float)
            and 0.0 <= dropout < 1.0
    ):
        raise RuntimeError("❌ 'model.params.dropout' must be a float within [0.0, 1.0).")
    if num_layers == 1 and dropout > 0:
        info("ℹ️ 'dropout' is ignored when 'num_layers' == 1.")

    # check proj_size integer in [0, hidden_size]
    if not (
            isinstance(proj_size, int)
            and 0 <= proj_size <= hidden_size
    ):
        raise RuntimeError("❌ 'model.params.proj_size' must be an integer in [0, hidden_size].")