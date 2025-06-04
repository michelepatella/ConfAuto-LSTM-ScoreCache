import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from utils.data.dataset_utils import split_training_set
from utils.log_utils import info, debug
from utils.data.dataloader_utils import create_data_loader, extract_targets_from_dataloader
from utils.model.model_utils import model_setup
from utils.training.training_utils import train_n_epochs


def _compute_time_series_cv(
        training_set,
        params,
        config_settings
):
    """
    Method to compute Time Series cross-validation.
    :param training_set: The training set on which to
    perform the time series cross-validation.
    :param params: The hyperparameters of the model.
    :param config_settings: The configuration settings.
    :return: The final average loss.
    """
    # initial message
    info("🔄 Time Series Cross-Validation started...")

    try:
        # get the no. of samples in the dataset
        n_samples = len(training_set)

        # debugging
        debug(f"⚙️ No. of samples in the training set: {n_samples}.")

        # setup for Time Series Split
        tscv = TimeSeriesSplit(
            n_splits=config_settings.cv_num_folds
        )
    except (
            ValueError,
            TypeError
    ) as e:
        raise RuntimeError(f"❌ Error while instantiating Time Series Split: {e}.")

    fold_losses = []
    # iterate over the training set
    for train_idx, val_idx in tscv.split(
            np.arange(n_samples)
    ):

        # debugging
        debug(f"⚙️ Training idx (Time series CV): {train_idx}.")
        debug(f"⚙️ Validation idx (Time series CV): {val_idx}.")

        # define training and validation sets
        training_dataset, validation_dataset = split_training_set(
            training_set,
            config_settings,
            training_indices=train_idx,
            validation_indices=val_idx
        )

        # debugging
        debug(f"⚙️ Training size (Time series CV): {len(training_dataset)}.")
        debug(f"⚙️ Validation size (Time series CV): {len(validation_dataset)}.")

        # create training and validation loaders
        training_loader = create_data_loader(
            training_dataset,
            config_settings.training_batch_size,
            True
        )
        validation_loader = create_data_loader(
            validation_dataset,
            config_settings.training_batch_size,
            False
        )

        try:
            # setup for training
            device, criterion, model, optimizer = model_setup(
                params["model"]["params"],
                params["training"]["optimizer"]["learning_rate"],
                extract_targets_from_dataloader(training_loader),
                config_settings
            )
        except (
            KeyError,
            TypeError,
            AttributeError,
            ValueError
        ) as e:
            raise RuntimeError(f"❌ Error while performing setup for training: {e}.")

        # train the model
        avg_loss, _ = train_n_epochs(
            config_settings.validation_num_epochs,
            model,
            training_loader,
            optimizer,
            criterion,
            device,
            config_settings,
            validation_loader=validation_loader,
            early_stopping=True
        )

        try:
            # save avg loss
            if avg_loss is not None:
                fold_losses.append(avg_loss)
        except (
            NameError,
            AttributeError,
            TypeError
        ) as e:
            raise RuntimeError(f"❌ Error while saving average loss: {e}.")

    # show a successful message
    info("🟢 Time Series Cross-Validation completed.")

    try:
        # calculate the average of loss
        final_avg_loss = np.mean(fold_losses)
    except (
        NameError,
        TypeError,
        ValueError
    ) as e:
        raise RuntimeError(f"❌ Error while calculating final average loss: {e}.")

    return final_avg_loss