import logging
from utils.AccessLogsDataset import AccessLogsDataset
from utils.config_utils import _get_config_value
from utils.dataset_utils import _create_data_loader, _get_dataset_path_type
from utils.training_utils import _training_setup, _save_trained_model, _train_n_epochs


def training():
    """
    Method to train the LSTM model.
    :return:
    """
    # initial message
    logging.info("🔄 Training started...")

    # get the dataset path
    dataset_path, _ = _get_dataset_path_type()

    # load the training set
    dataset = AccessLogsDataset(
        _get_config_value(dataset_path),
    "training"
    )

    # create the training loader
    training_loader = _create_data_loader(
        dataset,
        _get_config_value("training.batch_size")
    )

    # setup for training
    device, criterion, model, optimizer = (
        _training_setup(
            _get_config_value("model.params"),
            _get_config_value("training.learning_rate"),
        )
    )

    # train the model
    _train_n_epochs(
        _get_config_value("training.epochs"),
        model,
        training_loader,
        optimizer,
        criterion,
        device
    )

    # save the trained model
    _save_trained_model(model)

    # print a successful message
    logging.info("✅ Training successfully completed.")