import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import TimeSeriesSplit
from config.main import num_folds, training_batch_size, validation_epochs
from utils.log_utils import _info, _debug
from utils.data_utils import _create_data_loader, _extract_targets_from_loader
from utils.evaluation_utils import _evaluate_model
from utils.model_utils import _model_setup
from utils.training_utils import _train_n_epochs


def _time_series_cv(training_set, params):
    """
    Method to perform Time Series cross-validation.
    :param params: The hyperparameters of the model.
    :return: The final average loss.
    """
    # initial message
    _info("🔄 Time Series Cross-Validation started...")

    # get the no. of samples in the dataset
    n_samples = len(training_set)

    # debugging
    _debug(f"⚙️ No. of samples in the training set: {n_samples}.")

    try:
        # setup for Time Series Split
        tscv = TimeSeriesSplit(n_splits=num_folds)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"❌ Error while instantiating Time Series Split: {e}")

    fold_losses = []
    # iterate over the training set
    for train_idx, val_idx in tscv.split(np.arange(n_samples)):

        # debugging
        _debug(f"⚙️ Training idx (Time series CV): {train_idx}.")
        _debug(f"⚙️ Validation idx (Time series CV): {val_idx}.")

        try:
            # define training and validation sets
            training_dataset = Subset(training_set, train_idx)
            validation_dataset = Subset(training_set, val_idx)
        except (TypeError, IndexError, ValueError, AttributeError) as e:
            raise RuntimeError(f"❌ Error while defining training and validation sets: {e}")

        # debugging
        _debug(f"⚙️ Training size (Time series CV): {len(training_dataset)}.")
        _debug(f"⚙️ Validation size (Time series CV): {len(validation_dataset)}.")

        # create training and validation loaders
        training_loader = _create_data_loader(
            training_dataset,
            training_batch_size,
            True
        )
        validation_loader = _create_data_loader(
            validation_dataset,
            training_batch_size,
            False
        )

        # setup for training
        device, criterion, model, optimizer = _model_setup(
            params["model"]["params"],
            params["training"]["learning_rate"],
            _extract_targets_from_loader(training_loader)
        )

        # train the model
        _train_n_epochs(
            validation_epochs,
            model,
            training_loader,
            optimizer,
            criterion,
            device,
            validation_loader=validation_loader,
            early_stopping=True
        )

        # evaluate the model
        avg_loss, _, _ = _evaluate_model(
            model,
            validation_loader,
            criterion,
            device
        )
        fold_losses.append(avg_loss)

    # show a successful message
    _info("🟢 Time Series Cross-Validation completed.")

    # calculate the average of loss
    final_avg_loss = np.mean(fold_losses)

    return final_avg_loss