from tqdm import tqdm
from training.utils.backward_runner import compute_backward
from utils.logs.log_utils import info
from utils.model.forward.forward_runner import compute_forward


def train_one_epoch(
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
    info("🔄 Epoch training started...")

    # to show the progress bar
    training_loader = tqdm(
        training_loader,
        desc="🧠 Training Progress",
        leave=False
    )
    try:
        model.train()

        for x_features, x_keys, y_key in training_loader:
            # reset the gradients
            optimizer.zero_grad()

            # forward pass
            loss, _ = compute_forward(
                (x_features, x_keys, y_key),
                model,
                criterion,
                device
            )

            # check loss
            if loss is None:
                raise ValueError("Error while training the model due to None loss returned.")

            # backward pass
            compute_backward(
                loss,
                optimizer
            )

            training_loader.set_postfix(
                loss=loss.item()
            )
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except StopIteration as e:
        raise StopIteration(f"StopIteration: {e}.")
    except AssertionError as e:
        raise AssertionError(f"AssertionError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Epoch training completed.")