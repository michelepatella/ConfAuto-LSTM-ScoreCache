import logging


def _check_and_update_best_params(
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
    logging.info("🔄 Best parameters check and update started...")

    # if the average loss is less than the best one,
    # update it and the best parameters
    if (best_avg_loss is not None
            and best_avg_loss >= 0
            and avg_loss is not None):
        if avg_loss < best_avg_loss:
            # update the best loss
            best_avg_loss = avg_loss

            # update the best parameters
            best_params = curr_params

            # print updated parameters and best average loss
            logging.info(f"🆕 Updated best parameters: {best_params['model']} {best_params['training']}")
            logging.info(f"🆕 Updated best average loss: {best_avg_loss}")
    else:
        raise Exception(f"❌ Invalid best average loss ({best_avg_loss}) or average loss ({avg_loss}).")

    # print a successful message
    logging.info("🟢 Best parameters check and update completed.")

    return best_avg_loss, best_params