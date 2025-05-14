import logging
from utils.config_utils import _get_config_value
from utils.setup_utils import _training_testing_setup, _loader_setup
from utils.training_utils import _save_trained_model, _train_n_epochs


def training():
    """
    Method to train the LSTM model.
    :return:
    """
    # initial message
    logging.info("🔄 Training started...")

    # dataloader setup
    training_loader = _loader_setup("training")

    # setup for training
    device, criterion, model, optimizer = (
        _training_testing_setup(
            _get_config_value("model.params"),
            _get_config_value("training.learning_rate")
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