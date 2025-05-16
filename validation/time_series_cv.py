import numpy as np
import logging
from torch.utils.data import Subset
from sklearn.model_selection import TimeSeriesSplit
from utils.config_utils import _get_config_value
from utils.dataset_utils import _create_data_loader
from utils.evaluation_utils import _evaluate_model
from utils.setup_utils import _training_testing_setup, _extract_targets_from_loader
from utils.training_utils import _train_n_epochs


def _time_series_cv(training_set, params):
    """
    Method to perform Time Series cross-validation.
    :param params: The hyperparameters of the model.
    :return: The final average loss.
    """
    # initial message
    logging.info("🔄 Time Series started...")

    # get the no. of samples in the dataset
    n_samples = len(training_set)

    # debugging
    logging.debug(f"⚙️ No. of samples in the training set: {n_samples}.")

    try:
        # setup for Time Series Split
        tscv = TimeSeriesSplit(n_splits=_get_config_value(
            "validation.num_folds"
        ))
    except Exception as e:
        raise Exception(f"❌ Error while instantiating Time Series Split: {e}")

    fold_losses = []
    # iterate over the training set
    for train_idx, val_idx in tscv.split(np.arange(n_samples)):

        try:
            # define training and validation sets
            training_dataset = Subset(training_set, train_idx)
            validation_dataset = Subset(training_set, val_idx)
        except Exception as e:
            raise Exception(f"❌ Error while defining training and validation sets: {e}")

        # create training and validation loaders
        training_loader = _create_data_loader(
            training_dataset,
            _get_config_value("training.batch_size"),
            True
        )
        validation_loader = _create_data_loader(
            validation_dataset,
            _get_config_value("training.batch_size"),
            False
        )

        # setup for training
        device, criterion, model, optimizer = _training_testing_setup(
            params["model"]["params"],
            params["training"]["learning_rate"],
            _extract_targets_from_loader(training_loader)
        )

        # train the model
        _train_n_epochs(
            _get_config_value("validation.epochs"),
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

    logging.info("🟢 Time Series CV completed.")

    # calculate the average of loss
    final_avg_loss = np.mean(fold_losses)

    return final_avg_loss