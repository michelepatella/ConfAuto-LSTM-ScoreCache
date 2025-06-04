import torch
from utils.logs.log_utils import info
from utils.model.forward.forward_runner import compute_forward
from utils.model.forward.mc.mc_dropout_activator import enable_mc_dropout


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
        enable_mc_dropout(model)
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
                    _, outputs = compute_forward(
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