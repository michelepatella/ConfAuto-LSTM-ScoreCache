import logging
import numpy as np


def _check_and_update_best_params(
        fold_losses,
        best_avg_loss,
        curr_params,
        best_params
):
    """
    Method to calculate the average loss and update the best parameters
    in case the average loss is less than the current best loss.
    :param fold_losses: The loss of the current fold iteration.
    :param best_avg_loss: The current best average loss.
    :param curr_params: The current parameters (used in the current fold iteration).
    :param best_params: The current best parameters.
    :return: The best average loss and the best parameters.
    """
    if not fold_losses:
        raise Exception("Error checking and updating best parameters due to empty fold losses.")

    try:
        # calculate the average loss
        avg_loss = np.mean(fold_losses)
    except Exception as e:
        raise Exception(f"Error while calculating the average loss: {e}")

    # if the average loss is less than the best one,
    # update it and the best params
    if best_avg_loss is not None and best_avg_loss >= 0:
        if avg_loss < best_avg_loss:
            # update the best loss
            best_avg_loss = avg_loss

            try:
                # update the best params
                best_params = {
                    "hidden_size": curr_params["hidden_size"],
                    "num_layers": curr_params["num_layers"],
                    "dropout": curr_params["dropout"],
                    "learning_rate": curr_params["learning_rate"]
                }

                # print updated parameters and best average loss
                logging.info(f"Updated best parameters: {best_params}")
                logging.info(f"Updated best average loss: {best_avg_loss}")

            except Exception as e:
                raise Exception(f"Error while updating the best parameters: {e}")
    else:
        raise Exception(f"Invalid average loss: {avg_loss}")

    return best_avg_loss, best_params