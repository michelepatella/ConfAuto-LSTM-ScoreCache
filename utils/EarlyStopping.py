from utils.config_utils import _get_config_value
import numpy as np


class EarlyStopping:

    def __init__(self):
        """
        Method to initialize the class.
        """
        try:
            # set the fields
            self.patience = _get_config_value("early_stopping.patience")
            self.delta = _get_config_value("early_stopping.delta")
            self.best_avg_top_k_accuracy = -np.inf
            self.counter = 0
            self.early_stop = False
        except Exception as e:
            raise Exception(f"❌ Error while setting Early Stopping object's fields: {e}")


    def __call__(self, avg_top_k_accuracy):
        """
        Method called whenever Early Stopping object is invoked.
        :param avg_top_k_accuracy: The average top-k accuracy value.
        :return: 
        """
        try:
            # check whether the new avg top-k accuracy is greater
            # than the current best one
            if avg_top_k_accuracy > self.best_avg_top_k_accuracy + self.delta:
                # update the new best avg top-k accuracy
                # and reset the counter to trigger early stopping
                self.best_avg_top_k_accuracy = avg_top_k_accuracy
                self.counter = 0
            else:
                # increment the counter to trigger early stopping
                self.counter += 1

                # check whether the counter exceeds the patience
                if self.counter >= self.patience:
                    # early stopping is triggered
                    self.early_stop = True
        except Exception as e:
            raise Exception(f"❌ Error while calling Early Stopping's object: {e}")