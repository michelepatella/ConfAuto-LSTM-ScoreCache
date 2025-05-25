from torch.utils.data import Subset
from utils.AccessLogsDataset import AccessLogsDataset
from utils.log_utils import info, phase_var
from utils.dataloader_utils import dataloader_setup, extract_targets_from_dataloader, create_data_loader
from utils.training_utils import train_n_epochs
from utils.model_utils import save_model, model_setup


def training(config_settings):
    """
    Method to train the LSTM model.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Training started...")

    # set the variable indicating the state of the process
    phase_var.set("training")

    # dataloader setup
    training_set, _ = dataloader_setup(
        "training",
        config_settings.training_batch_size,
        False,
        config_settings,
        AccessLogsDataset
    )

    # calculate total training set size
    total_training_size = len(training_set)

    # calculate training and validation size
    training_size = int(
        (1.0 - config_settings.validation_perc) * total_training_size
    )
    validation_size = int(
        config_settings.validation_perc * total_training_size
    )

    # create indexes for training and validation
    training_indices = list(range(
        0,
        training_size
    ))
    validation_indices = list(range(
        training_size,
        training_size + validation_size
    ))

    # split the training set into training and validation set
    final_training_set = Subset(
        training_set,
        training_indices
    )
    final_validation_set = Subset(
        training_set,
        validation_indices
    )

    # create a loader for each set
    training_loader = create_data_loader(
        final_training_set,
        config_settings.training_batch_size,
        True
    )
    validation_loader = create_data_loader(
        final_validation_set,
        config_settings.training_batch_size,
        False
    )

    # setup for training
    device, criterion, model, optimizer = (
        model_setup(
            config_settings.model_params,
            config_settings.learning_rate,
            extract_targets_from_dataloader(training_loader),
            config_settings
        )
    )

    # train the model
    _, model = train_n_epochs(
        config_settings.training_num_epochs,
        model,
        training_loader,
        optimizer,
        criterion,
        device,
        config_settings,
        early_stopping=True,
        validation_loader=validation_loader
    )

    # save the best model trained
    save_model(model, config_settings)

    # print a successful message
    info("✅ Training successfully completed.")