from utils.logs.log_utils import info


def compute_backward(
        loss,
        optimizer
):
    """
    Method to compute backward pass.
    :param loss: The loss to back propagate.
    :param optimizer: The optimizer to use.
    :return:
    """
    # initial message
    info("🔄 Backward pass started...")

    try:
        # backward pass
        loss.backward()

        # optimize backward pass
        optimizer.step()
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Backward pass computed.")