from utils.logs.log_utils import info, debug


def check_and_update_best_params(
        avg_loss,
        best_avg_loss,
        curr_params,
        best_params,
):
    """
    Method to calculate the average loss and update the best parameters
    in case the average loss is less than the current best one.
    :param avg_loss: The loss of the current fold iteration.
    :param best_avg_loss: The current best average loss.
    :param curr_params: The current parameters (used in the current fold iteration).
    :param best_params: The best parameters found so far.
    :return: The best average loss and the best parameters.
    """
    # initial message
    info("🔄 Best parameters check and update started...")

    # debugging
    debug(f"⚙️ Avg loss: {avg_loss}.")
    debug(f"⚙️ Best avg loss: {best_avg_loss}.")

    try:
        # check avg loss and best avg loss
        if (
            avg_loss is None or
            not isinstance(avg_loss, (float, int))
        ):
            raise ValueError(f" avg_loss must be a float or int. Received: {avg_loss}.")
        if (
            best_avg_loss is not None and
            not isinstance(best_avg_loss, (float, int))
        ):
            raise ValueError(f" best_avg_loss must be a float or int. Received: {best_avg_loss}.")

        # if the average loss is less than the best one,
        # update it and the best parameters
        if (
            best_avg_loss is not None and
            best_avg_loss >= 0 and
            avg_loss is not None and
            avg_loss < best_avg_loss
        ):

            # update the best loss
            best_avg_loss = avg_loss

            # update the best parameters
            best_params = curr_params

            # print updated parameters and best average loss
            info(f"🆕 Updated best parameters: {best_params['model']}"
                  f" {best_params['training']}")
            info(f"🆕 Updated best average loss: {best_avg_loss}")
        else:
            info("ℹ️ No best average loss improvement, best "
                  "parameters not updated.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")

    # print a successful message
    info("🟢 Best parameters check and update completed.")

    return best_avg_loss, best_params