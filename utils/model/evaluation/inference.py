import torch
from utils.logs.log_utils import info, debug
from utils.model.forward.mc.mc_forward_runner import mc_forward_passes


def infer_batch(
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
    (
        all_preds,
        all_targets,
        all_outputs
    ) = [], [], []
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

                (
                    outputs_mean,
                    outputs_var, _) = mc_forward_passes(
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
                    raise ValueError("Error while computing average loss due to loss equals None.")

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
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Batch inferred.")

    return (
        total_loss,
        all_preds,
        all_targets,
        all_outputs,
        all_vars
    )