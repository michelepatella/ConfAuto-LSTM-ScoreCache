from utils.feedfarward_utils import _compute_forward, _compute_backward


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
    # train the model
    model.train()

    for x, y in training_loader:

        try:
            # reset the gradients
            optimizer.zero_grad()
        except Exception as e:
            raise Exception(f"Error resetting the gradients: {e}")

        # forward pass
        loss, _ = _compute_forward((x, y), model, criterion, device)

        # check loss
        if loss is None:
            raise Exception("Error while training the model due to None loss returned.")

        # backward pass
        _compute_backward(loss, optimizer)