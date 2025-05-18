import torch
from sympy.physics.units import momentum
from torch.cuda import CudaError

from main import config_settings
from utils.log_utils import _info, _debug
from utils.EarlyStopping import EarlyStopping
from utils.evaluation_utils import _evaluate_model
from utils.feedforward_utils import _compute_forward, _compute_backward


def _train_one_epoch(
        model,
        training_loader,
        optimizer,
        criterion,
        device
):
    """
    Method to train the model one epoch.
    :param model: Model to be trained.
    :param training_loader: Training data loader.
    :param optimizer: Optimizer to be used.
    :param criterion: The loss function.
    :param device: Device to be used.
    :return:
    """
    # initial message
    _info("🔄 Epoch training started...")

    model.train()

    # to show the progress bar
    """training_loader = tqdm(
        training_loader,
        desc="🧠 Training Progress",
        leave=False
    )
    """
    try:
        for x_features, x_keys, y_key in training_loader:
            # reset the gradients
            optimizer.zero_grad()

            # forward pass
            loss, _ = _compute_forward(
                (x_features, x_keys, y_key),
                model,
                criterion,
                device
            )

            # check loss
            if loss is None:
                raise ValueError("❌ Error while training the model due to None loss returned.")

            # backward pass
            _compute_backward(loss, optimizer)

            # training_loader.set_postfix(loss=loss.item())

    except (AttributeError, TypeError, ValueError, StopIteration, CudaError, AssertionError) as e:
        raise RuntimeError(f"❌ Error while training the model (one-epoch): {e}.")

    # show a successful message
    _info("🟢 Epoch training completed.")


def _train_n_epochs(
        epochs,
        model,
        training_loader,
        optimizer,
        criterion,
        device,
        early_stopping=False,
        validation_loader=None,
):
    """
    Method to train the model a specified number of epochs.
    :param epochs: Number of epochs.
    :param model: The model to be trained.
    :param training_loader: The training loader.
    :param optimizer: The optimizer to be used.
    :param criterion: The loss function.
    :param device: The device to be used.
    :param early_stopping: Whether to apply early stopping or not.
    :param validation_loader: Validation data loader.
    :return:
    """
    # debugging
    _debug(f"⚙️ Number of epochs: {epochs}.")
    _debug(f"⚙️ Training loader size: {len(training_loader)}.")
    _debug(f"⚙️ Optimizer to use: {optimizer}.")
    _debug(f"⚙️ Criterion to use: {criterion}.")
    _debug(f"⚙️ Device to use: {device}.")
    _debug(f"⚙️ Early stopping: {'Enabled' if early_stopping else 'Disabled'}.")
    _debug(f"⚙️ Validation loader: {'Received' if validation_loader is not None else 'Not received'}.")

    try:
        es = None
        # instantiate early stopping object (if needed)
        if early_stopping:
            es = EarlyStopping()

        # n-epochs learning
        for epoch in range(epochs):
            _info(f"⏳ Epoch {epoch + 1}/{epochs}")

            # train the model
            _train_one_epoch(
                model,
                training_loader,
                optimizer,
                criterion,
                device
            )

            if early_stopping:

                avg_loss = None
                if validation_loader:

                    # get the validation average loss
                    avg_loss, _,  _ = _evaluate_model(
                        model,
                        validation_loader,
                        criterion,
                        device
                    )

                # early stopping logic
                if early_stopping and avg_loss is not None:
                    es(avg_loss)
                    # check whether to stop
                    if es.early_stop:
                        _info("🛑 Early stopping triggered.")
                        break

    except (NameError, AttributeError, TypeError, ValueError, CudaError, LookupError) as e:
        raise RuntimeError(f"❌ Error while training the model (n-epochs): {e}.")


def _build_optimizer(model, learning_rate):
    """
    Method to build the optimizer.
    :param model: Model for which the optimizer will be built.
    :param learning_rate: Learning rate.
    :return: The created optimizer.
    """
    # initial message
    _info("🔄 Optimizer building started...")

    # debugging
    _debug(f"⚙️ Learning rate: {learning_rate}.")
    _debug(f"⚙️ Optimizer type: {config_settings["optimizer_type"]}.")

    try:
        # define the optimizer
        if config_settings["optimizer_type"] == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )
        elif config_settings["optimizer_type"] == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=config_settings["weight_decay"]
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum
            )
    except (ValueError, TypeError, UnboundLocalError, KeyError, AssertionError) as e:
        raise RuntimeError(f"❌ Error while building optimizer: {e}.")

    # show a successful message
    _info("🟢 Optimizer building completed.")

    return optimizer