from tqdm import tqdm
import logging
import torch
from utils.config_utils import _get_config_value
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
    logging.info("🔄 Epoch training started...")

    model.train()

    # to show the progress bar
    training_loader = tqdm(
        training_loader,
        desc="🧠 Training Progress",
        leave=False
    )

    for x, y in training_loader:
        try:
            # reset the gradients
            optimizer.zero_grad()
        except Exception as e:
            raise Exception(f"❌ Error resetting the gradients: {e}")

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

        training_loader.set_postfix(loss=loss.item())

    # show a successful message
    logging.info("🟢 Epoch training completed.")


def _build_optimizer(model, learning_rate):
    """
    Method to build the optimizer.
    :param model: Model for which the optimizer will be built.
    :param learning_rate: Learning rate.
    :return: The created optimizer.
    """
    # read the optimizer
    optimizer = _get_config_value("training.optimizer")

    try:
        # define the optimizer
        if optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )
        elif optimizer == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=_get_config_value("training.weight_decay")
            )
        elif optimizer == "rmsprop":
            return torch.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                momentum=_get_config_value("training.momentum")
            )
        elif optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=_get_config_value("training.momentum")
            )
        else:
            raise Exception(f"❌ Invalid optimizer: {optimizer}")
    except Exception as e:
        raise Exception(f"❌ Error while building optimizer: {e}")