import torch
from utils.log_utils import _info, _debug
from utils.EarlyStopping import EarlyStopping
from utils.config_utils import _get_config_value
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
    for x, y in training_loader:
        try:
            # reset the gradients
            optimizer.zero_grad()
        except Exception as e:
            raise Exception(f"❌ Error while resetting the gradients: {e}")

        # forward pass
        loss, _ = _compute_forward(
            (x, y),
            model,
            criterion,
            device
        )

        # check loss
        if loss is None:
            raise Exception("❌ Error while training the model due to None loss returned.")

        # backward pass
        _compute_backward(loss, optimizer)

        # training_loader.set_postfix(loss=loss.item())

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
    _debug(f"⚙️ Early Stopping: {early_stopping}.")
    _debug(f"⚙️ Validation loader: {'Received' if validation_loader is not None else 'Not received'}.")

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


def _build_optimizer(model, learning_rate):
    """
    Method to build the optimizer.
    :param model: Model for which the optimizer will be built.
    :param learning_rate: Learning rate.
    :return: The created optimizer.
    """
    # initial message
    _info("🔄 Optimizer building started...")

    # read the optimizer
    optimizer_type = _get_config_value("training.optimizer")

    # debugging
    _debug(f"⚙️ Learning rate: {learning_rate}.")
    _debug(f"⚙️ Optimizer type: {optimizer_type}.")

    try:
        # define the optimizer
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=_get_config_value("training.weight_decay")
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=_get_config_value("training.momentum")
            )
        else:
            raise Exception(f"❌ Invalid optimizer: {optimizer_type}")
    except Exception as e:
        raise Exception(f"❌ Error while building optimizer: {e}")

    # show a successful message
    _info("🟢 Optimizer building completed.")

    return optimizer