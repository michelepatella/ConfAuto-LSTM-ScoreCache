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
            if isinstance(module, torch.nn.Dropout):
                module.train()
        # enable dropout output
        model.use_mc_dropout = True
    except (AttributeError, TypeError) as e:
        raise RuntimeError(f"❌ Error while inferring the batch: {e}.")

    # show a successful message
    info("🟢 MC dropout enabled.")


def _infer_batch(
        model,
        loader,
        criterion,
        device,
        mc_dropout_samples=1
):
    """
    Method to infer the batch.
    :param model: The model to be used.
    :param loader: The dataloader.
    :param criterion: The loss function.
    :param device: The device to be used.
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

                # if there is more than one MC sample
                # enable MC dropout
                if mc_dropout_samples > 1:
                    _enable_mc_dropout(model)

                outputs_mc = []
                # for each MC sample
                for _ in range(mc_dropout_samples):

                    # calculate loss and outputs through forward pass
                    _, outputs = _compute_forward(
                        (x_features, x_keys, y_key),
                        model,
                        None,
                        device
                    )

                    # store the output
                    outputs_mc.append(outputs.unsqueeze(0))

                # save the mean of the outputs of the model
                # for a specific input
                outputs_mc_tensor = torch.cat(outputs_mc, dim=0)
                outputs_mean = outputs_mc_tensor.mean(dim=0)

                if mc_dropout_samples > 1:
                    # obtain and save the variance
                    outputs_var = outputs_mc_tensor.var(
                        dim=0,
                        unbiased=False
                    )
                    all_vars.extend(outputs_var.cpu())

                # compute the loss
                loss = criterion(outputs_mean, y_key)

                # debugging
                debug(f"⚙️ Loss computed: {loss}.")
                debug(f"⚙️ Outputs shape: {outputs.shape}.")

                # check the loss
                if loss is None:
                    raise ValueError("❌ Error while computing average loss due to loss equals None.")

                # update the total loss
                total_loss += loss.item()

                # store predictions and target for metrics
                preds = torch.argmax(outputs_mean, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_key.cpu().numpy())
                all_outputs.extend(outputs_mean.cpu())

    except (IndexError, ValueError, KeyError, AttributeError, TypeError) as e:
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
        z_score = norm.ppf(1 - (1 - config_settings.confidence_level) / 2)

        # debugging
        debug(f"⚙️ Z-score for CIs: {z_score}.")

        # calculate the standard deviation
        outputs_std = torch.sqrt(torch.stack(all_vars))

        # calculate lower and upper CIs boundaries
        lower_ci = torch.stack(all_outputs) - z_score * outputs_std
        upper_ci = torch.stack(all_outputs) + z_score * outputs_std

    except (NameError, TypeError, ValueError, IndexError) as e:
        raise RuntimeError(f"❌ Error while calculating "
                           f"confidence intervals: {e}.")

    # show a successful message
    info("🟢 Confidence intervals calculated.")

    return lower_ci, upper_ci


def autoregressive_rollout(
    model,
    seed_sequence,
    config_settings,
    rollout_steps,
    device
):

    model.train()

    x_features_seq, x_keys_seq, _ = seed_sequence
    x_features_seq = x_features_seq.unsqueeze(0).to(device)
    x_keys_seq = x_keys_seq.unsqueeze(0).to(device)

    all_outputs = []
    all_vars = []

    for _ in range(rollout_steps):
        mc_outputs = []

        # MC samples
        for _ in range(config_settings.mc_dropout_num_samples):
            output_logits = model(x_features_seq, x_keys_seq)  # [1, num_classes]
            mc_outputs.append(output_logits.detach())

        mc_stack = torch.stack(mc_outputs)  # [num_samples, 1, num_classes]
        mean_logits = mc_stack.mean(dim=0).squeeze(0)  # [num_classes]
        var_logits = mc_stack.var(dim=0).squeeze(0)    # [num_classes]

        all_outputs.append(mean_logits)
        all_vars.append(var_logits)

        # argmax del logit medio => nuovo token
        pred_key = mean_logits.argmax().unsqueeze(0).unsqueeze(0)  # [1, 1]

        # aggiorna la sequenza chiavi
        x_keys_seq = torch.cat([x_keys_seq[:, 1:], pred_key], dim=1)
        # Nota: puoi anche aggiornare x_features_seq se necessario

    return all_outputs, all_vars
