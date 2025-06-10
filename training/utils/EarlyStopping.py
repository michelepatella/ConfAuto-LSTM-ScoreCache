import numpy as np
from utils.logs.log_utils import debug, info, phase_var


class EarlyStopping:

    def __init__(
            self,
            config_settings
    ):
        """
        Method to initialize the class.
        :param config_settings: The configuration settings.
        """
        # initial message
        info("🔄 EarlyStopping initialization started...")

        try:
            # read the current phase
            current_phase = phase_var.get()

            # set the fields
            self.patience = config_settings.validation_early_stopping_patience \
                if current_phase == "validation" \
                else config_settings.training_early_stopping_patience
            self.delta = config_settings.validation_early_stopping_delta \
                if current_phase == "validation" \
                else config_settings.training_early_stopping_delta
            self.best_avg_loss = np.inf
            self.counter = 0
            self.early_stop = False
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # debugging
        debug(f"⚙️ Patience for Early Stopping: {self.patience}.")
        debug(f"⚙️ Delta for Early Stopping: {self.delta}.")
        debug(f"⚙️ Best avg loss: {self.best_avg_loss}.")

        # show a successful message
        info(f"🟢 EarlyStopping initialized.")


    def __call__(
            self,
            avg_loss
    ):
        """
        Method called whenever Early Stopping object is invoked.
        :param avg_loss: The average loss value.
        :return: 
        """
        # initial message
        info("🔄 EarlyStopping check started...")

        try:
            # check whether the avg loss is less
            # than the current best one
            if avg_loss < self.best_avg_loss - self.delta:
                # update the best avg loss
                # and reset the counter used to trigger early stopping
                self.best_avg_loss = avg_loss
                self.counter = 0

                # debugging
                debug(f"⚙️ New best average loss: {self.best_avg_loss}.")
            else:
                # increment the counter to trigger early stopping
                self.counter += 1

                # debugging
                debug(f"⚙️ Counter value updated: {self.counter}.")

                # check whether the counter exceeds the patience
                if self.counter >= self.patience:
                    # early stopping is triggered
                    self.early_stop = True
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

        # show a successful message
        info(f"🟢 EarlyStopping check completed.")