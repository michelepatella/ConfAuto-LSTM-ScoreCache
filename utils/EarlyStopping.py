import numpy as np
from utils.log_utils import _debug


class EarlyStopping:

    def __init__(self):
        """
        Method to initialize the class.
        """
        from main import config_settings

        try:
            # set the fields
            self.patience = config_settings.early_stopping_patience
            self.delta = config_settings.early_stopping_delta
            self.best_avg_loss = np.inf
            self.counter = 0
            self.early_stop = False
        except (NameError, AttributeError, TypeError) as e:
            raise RuntimeError(f"❌ Error setting the class fields: {e}.")

        # debugging
        _debug(f"⚙️ Patience for Early Stopping: {self.patience}.")
        _debug(f"⚙️ Delta for Early Stopping: {self.delta}.")
        _debug(f"⚙️ Best avg loss: {self.best_avg_loss}.")


    def __call__(self, avg_loss):
        """
        Method called whenever Early Stopping object is invoked.
        :param avg_loss: The average loss value.
        :return: 
        """
        try:
            # check whether the avg loss is less
            # than the current best one
            if avg_loss < self.best_avg_loss - self.delta:

                # update the best avg loss
                # and reset the counter used to trigger early stopping
                self.best_avg_loss = avg_loss
                self.counter = 0

                # debugging
                _debug(f"⚙️ New best average loss: {self.best_avg_loss}.")

            else:

                # increment the counter to trigger early stopping
                self.counter += 1

                # debugging
                _debug(f"⚙️ Counter value updated: {self.counter}.")

                # check whether the counter exceeds the patience
                if self.counter >= self.patience:
                    # early stopping is triggered
                    self.early_stop = True

        except (AttributeError, TypeError, NameError) as e:
            raise RuntimeError(f"❌ Error while calling Early Stopping's object: {e}.")