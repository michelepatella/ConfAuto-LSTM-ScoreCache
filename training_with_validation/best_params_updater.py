import logging
import numpy as np


def _check_and_update_best_params(
        avg_loss,
        best_avg_loss,
        curr_params,
        best_params,
):
    """
    Method to calculate the average loss and update the best parameters
    in case the average loss is less than the current best loss.
    :param avg_loss: The loss of the current fold iteration.
    :param best_avg_loss: The current best average loss.
    :param curr_params: The current parameters (used in the current fold iteration).
    :param best_params: The best parameters found so far.
    :return: The best average loss and the best parameters.
    """
    # initial message
    logging.info("🔄 Best parameters check and update started...")

    best_params = best_params

    # if the average loss is less than the best one,
    # update it and the best parameters
    if best_avg_loss is not None and best_avg_loss >= 0\
            and avg_loss is not None:
        if avg_loss < best_avg_loss:
            # update the best loss
            best_avg_loss = avg_loss

            try:
                # update the best parameters
                best_params = {
                    "model":
                        {
                            "hidden_size": curr_params["model"]["hidden_size"],
                            "num_layers": curr_params["model"]["num_layers"],
                            "dropout": curr_params["model"]["dropout"]
                        },
                    "training":
                        {
                            "learning_rate": curr_params["training"]["learning_rate"]
                        }
                }

                # print updated parameters and best average loss
                logging.info(f"🆕 Updated best parameters: {best_params["model"]} {best_params["training"]}")
                logging.info(f"🆕 Updated best average loss: {best_avg_loss}")

            except Exception as e:
                raise Exception(f"❌ Error while updating the best parameters: {e}")
    else:
        raise Exception(f"❌ Invalid best average loss ({avg_loss}) or loss ({avg_loss}).")

    # print a successful message
    logging.info("🟢 Best parameters check and update completed.")

    return best_avg_loss, best_params