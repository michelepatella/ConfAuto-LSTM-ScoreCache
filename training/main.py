from utils.AccessLogsDataset import AccessLogsDataset
from utils.dataset_utils import split_training_set
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
    training_set, training_loader = dataloader_setup(
        "training",
        config_settings.training_batch_size,
        False,
        config_settings,
        AccessLogsDataset
    )

    # split training set into training and validation sets
    (
        final_training_set,
        final_validation_set
    ) = split_training_set(
        training_set,
        config_settings
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
    save_model(
        model,
        config_settings
    )

    # print a successful message
    info("✅ Training successfully completed.")