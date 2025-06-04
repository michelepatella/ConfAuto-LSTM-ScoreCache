import torch

from utils.logs.log_utils import info, debug


def _compute_forward(
        batch,
        model,
        criterion,
        device
):
    """
    Method to compute forward pass and loss based on batch of data.
    :param batch: The batch of data to process.
    :param model: The model to use.
    :param criterion: The loss function.
    :param device: The device to use.
    :return: The loss function (if computed) and the outputs for the batch.
    """
    # initial message
    info("🔄 Forward pass started...")

    # try unpack data
    try:
        # unpack data
        x_features, x_keys, y_key = batch

        # debugging
        debug(f"⚙️ Target batch: {y_key}.")
    except (
            ValueError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while error unpacking data: {e}.")

    try:
        # move data to the device
        x_features = x_features.to(device)
        x_keys = x_keys.to(device)
        y_key = y_key.to(device)
    except (
            AttributeError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while moving data to device: {e}.")

    try:
        # calculate the outputs
        outputs = model(
            x_features,
            x_keys
        )

        # debugging
        debug(f"⚙️ Input batch shape: {x_features.shape}.")
        debug(f"⚙️ Input keys shape: {x_keys.shape}")
        debug(f"⚙️ Model output shape: {outputs.shape}.")
    except (
            TypeError,
            AttributeError
    ) as e:
        raise RuntimeError(f"❌ Error during model inference: {e}.")

    loss = None
    if criterion is not None:
        try:
            # calculate the loss and update the total one
            loss = criterion(
                outputs,
                y_key
            )

            # debugging
            debug(f"⚙️ Loss: {loss.item()}.")
        except (
                TypeError,
                ValueError
        ) as e:
            raise RuntimeError(f"❌ Error while calculating loss: {e}.")

    # show a successful message
    info("🟢 Forward pass computed.")

    return loss, outputs


def _enable_mc_dropout(model):
    """
    Method to enable MC dropout.
    :param model: The model for which to enable MC dropout.
    :return:
    """
    # initial message
    info("🔄 MC dropout enabling started...")

    try:
        for module in model.modules():
            if isinstance(
                    module,
                    torch.nn.Dropout
            ):
                module.train()
                # debugging
                debug(f"⚙️ Dropout enabled.")

        # set dropout enabled
        model.use_mc_dropout = True
    except (
            AttributeError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while inferring the batch: {e}.")

    # show a successful message
    info("🟢 MC dropout enabled.")


def mc_forward_passes(
        model,
        inputs,
        device,
        config_settings,
        mc_dropout_samples=1
):
    """
    Method to perform forward passes with MC dropout (if enabled).
    :param model: The model for which to perform forward passes.
    :param inputs: The inputs to the model.
    :param device: The device to be used.
    :param config_settings: The configuration settings.
    :param mc_dropout_samples: The number of MC dropout samples.
    :return: The mean of outputs, the output variance, and
    the output tensors.
    """
    # initial message
    info("🔄 MC forward passes started...")

    # if more than one MC dropout sample
    if mc_dropout_samples > 1:
        # enable dropout at inference time
        _enable_mc_dropout(model)
    else:
        model.eval()

    try:
        # infer for each MC dropout sample
        outputs_mc = []
        with (torch.no_grad()):
            for _ in range(mc_dropout_samples):
                # check the type of input before inferring
                if (
                    isinstance(inputs, tuple) and
                    len(inputs) == config_settings.num_features+1
                ):
                    _, outputs = _compute_forward(
                        inputs,
                        model,
                        None,
                        device
                    )
                else:
                    outputs = model(*inputs)

                # store the output
                outputs_mc.append(
                    outputs.unsqueeze(0)
                )

        # calculate tensor, mean, and variance of outputs
        outputs_mc_tensor = torch.cat(
            outputs_mc,
            dim=0
        )
        outputs_mean = outputs_mc_tensor.mean(dim=0)
        outputs_var = outputs_mc_tensor.var(
            dim=0,
            unbiased=False
        ) if mc_dropout_samples > 1 else None

    except (
            TypeError,
            AttributeError,
            IndexError,
            ValueError
    ) as e:
        raise ValueError(f"❌ Error while computing MC forward passes: {e}.")

    # show a successful message
    info("🟢 MC forward passes computed.")

    return (
            outputs_mean,
            outputs_var,
            outputs_mc_tensor
    )