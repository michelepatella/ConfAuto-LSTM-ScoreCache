import math
import torch
from scipy.stats import norm
from utils.feedforward_utils import _compute_forward
from utils.log_utils import info, debug


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


def _infer_batch(
        model,
        loader,
        criterion,
        device,
        config_settings,
        mc_dropout_samples=1
):
    """
    Method to infer the batch.
    :param model: The model to be used.
    :param loader: The dataloader.
    :param criterion: The loss function.
    :param device: The device to be used.
    :param config_settings: The configuration settings.
    :param mc_dropout_samples: The number of MC dropout
    samples (=1 means no MC dropout).
    :return: The total loss, all the predictions,
    all the targets, all the outputs returned by the model and the variances.
    """
    # initial message
    info("🔄 Batch inference started...")

    # debugging
    debug(f"⚙️ Input loader batch size: {len(loader)}.")
    debug(f"⚙️ MC dropout samples: {mc_dropout_samples}.")

    # initialize data
    total_loss = 0.0
    all_preds, all_targets, all_outputs = [], [], []
    all_vars = []

    model.eval()

    try:
        with torch.no_grad():
            for x_features, x_keys, y_key in loader:

                # debugging
                debug(f"⚙️ Batch x_features shape: {x_features.shape}.")
                debug(f"⚙️ Batch x_keys shape: {x_keys.shape}.")
                debug(f"⚙️ Batch y_key shape: {y_key.shape}.")

                # move features and key on device
                x_features = x_features.to(device)
                x_keys = x_keys.to(device)
                y_key = y_key.to(device)

                outputs_mean, outputs_var, _ = mc_forward_passes(
                    model,
                    (x_features, x_keys, y_key),
                    device,
                    config_settings,
                    mc_dropout_samples
                )

                if outputs_var is not None:
                    all_vars.extend(outputs_var.cpu())

                # compute the loss
                loss = criterion(
                    outputs_mean,
                    y_key
                )

                # debugging
                debug(f"⚙️ Loss computed: {loss}.")

                # check the loss
                if loss is None:
                    raise ValueError("❌ Error while computing average loss due to loss equals None.")

                # update the total loss
                total_loss += loss.item()

                # store predictions and target for metrics
                preds = torch.argmax(
                    outputs_mean,
                    dim=1
                )
                all_preds.extend(
                    preds.cpu().numpy()
                )
                all_targets.extend(
                    y_key.cpu().numpy()
                )
                all_outputs.extend(
                    outputs_mean.cpu()
                )

    except (
            IndexError,
            ValueError,
            KeyError,
            AttributeError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while inferring the batch: {e}.")

    # show a successful message
    info("🟢 Batch inferred.")

    return (
        total_loss,
        all_preds,
        all_targets,
        all_outputs,
        all_vars
    )


def calculate_confidence_intervals(
        all_outputs,
        all_vars,
        config_settings
):
    """
    Method to calculate confidence intervals.
    :param all_outputs: The outputs of the model.
    :param all_vars: The variances of the outputs.
    :param config_settings: Configuration settings.
    :return: The upper and lower confidence interval boundaries.
    """
    # initial message
    info("🔄 Confidence intervals calculation started...")

    try:
        # calculate the z-score
        z_score = norm.ppf(
            1 - (1 - config_settings.confidence_level) / 2
        )

        # debugging
        debug(f"⚙️ Z-score for CIs: {z_score}.")

        # calculate the standard deviation
        outputs_tensor = torch.stack(all_outputs)
        outputs_std = torch.sqrt(
            torch.stack(all_vars)
        )

        # calculate lower and upper CIs boundaries
        lower_ci = (
                outputs_tensor -
                z_score * outputs_std
        )
        upper_ci = (
            outputs_tensor +
            z_score * outputs_std
        )

    except (
            NameError,
            TypeError,
            ValueError,
            IndexError
    ) as e:
        raise RuntimeError(f"❌ Error while calculating confidence intervals: {e}.")

    # show a successful message
    info("🟢 Confidence intervals calculated.")

    return lower_ci, upper_ci


def autoregressive_rollout(
    model,
    seed_sequence,
    device,
    config_settings
):
    """
    Method to perform autoregressive rollout.
    :param model: The model to use.
    :param seed_sequence: The seed sequence.
    :param device: The device to be used.
    :param config_settings: The configuration settings.
    :return: All the outputs and the variances.
    """
    # initial message
    info("🔄 Autoregressive rollout started...")

    try:
        # prepare data
        x_features_seq, x_keys_seq, _ = (
            seed_sequence
        )
        x_features_seq = (
            x_features_seq.unsqueeze(0).to(device)
        )
        x_keys_seq = (
            x_keys_seq.unsqueeze(0).to(device)
        )
        all_outputs = []
        all_vars = []

        # initialize last time to current time
        last_sin = x_features_seq[0, -1, 0].item()
        last_cos = x_features_seq[0, -1, 1].item()
        last_time = (
                math.atan2(last_sin, last_cos)
                % (2 * math.pi)
        )

        # increase the temporal features (2 minutes)
        delta_t = (2 / 1440) * (2 * math.pi)

        # for each future sequence
        for i in range(config_settings.prediction_interval):
            # compute MC forward pass
            outputs_mean, outputs_var, _ = mc_forward_passes(
                model,
                (x_features_seq, x_keys_seq),
                device,
                config_settings,
                config_settings.mc_dropout_num_samples
            )

            # store outputs and variances
            all_outputs.append(
                outputs_mean.squeeze(0)
            )
            if outputs_var is not None:
                all_vars.append(
                    outputs_var.squeeze(0)
                )
            else:
                all_vars.append(torch.zeros_like(
                    outputs_mean.squeeze(0)
                ))

            # get the predicted key as the most probable one
            pred_key = (
                outputs_mean.argmax(dim=-1).unsqueeze(1)
            )

            # add a new step using the predicted key
            x_keys_seq = torch.cat(
                [x_keys_seq[:, 1:], pred_key],
                dim=1
            )

            # update features
            last_time = (
                    (last_time + delta_t) %
                    (2 * math.pi)
            )
            new_cos = math.cos(last_time)
            new_sin = math.sin(last_time)
            new_feature = torch.tensor(
                [[new_sin, new_cos]],
                device=device
            ).unsqueeze(0)
            x_features_seq = torch.cat(
                [x_features_seq[:, 1:, :],
                 new_feature],
                dim=1
            )

            # debugging
            debug(f"⚙️ Step: {i}.")
            debug(f"⚙️ New Prediction: {pred_key}.")
            debug(f"⚙️ At time: {last_time}.")
            debug(f"⚙️ Variance: {outputs_var}.")

    except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while performing autoregressive rollout: {e}.")

    # show a successful message
    info("🟢 Autoregressive rollout completed.")

    return all_outputs, all_vars