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
    except (
            AttributeError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error during backpropagation: {e}.")

    # show a successful message
    info("🟢 Backward pass computed.")