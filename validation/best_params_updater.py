import logging


def _check_and_update_best_params(
        avg_top_k_accuracy,
        best_avg_top_k_accuracy,
        curr_params,
        best_params,
):
    """
    Method to calculate the average top-k accuracy and update the best parameters
    in case the average top-k accuracy is greater than the current best one.
    :param avg_top_k_accuracy: The average top-k accuracy of the current fold iteration.
    :param best_avg_top_k_accuracy: The current best average top-k accuracy.
    :param curr_params: The current parameters (used in the current fold iteration).
    :param best_params: The best parameters found so far.
    :return: The best average top-k accuracy and the best parameters.
    """
    # initial message
    logging.info("🔄 Best parameters check and update started...")

    # if the average top-k accuracy is greater than the best one,
    # update it and the best parameters
    if best_avg_top_k_accuracy is not None and best_avg_top_k_accuracy >= 0\
            and avg_top_k_accuracy is not None:
        if avg_top_k_accuracy > best_avg_top_k_accuracy:
            # update the best avg top-k accuracy
            best_avg_top_k_accuracy = avg_top_k_accuracy

            # update the best parameters
            best_params = curr_params

            # print updated parameters and best top-k accuracy
            logging.info(f"🆕 Updated best parameters: {best_params['model']} {best_params['training']}")
            logging.info(f"🆕 Updated best top-k accuracy: {best_avg_top_k_accuracy}")
    else:
        raise Exception(f"❌ Invalid best average top-k accuracy ({best_avg_top_k_accuracy}) or average top-k accuracy ({avg_top_k_accuracy}).")

    # print a successful message
    logging.info("🟢 Best parameters check and update completed.")

    return best_avg_top_k_accuracy, best_params