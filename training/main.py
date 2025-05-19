from utils.log_utils import _info, phase_var
from utils.dataloader_utils import _loader_setup, _extract_targets_from_loader
from utils.training_utils import _train_n_epochs
from utils.model_utils import _save_model, _model_setup


def training(config_settings):
    """
    Method to train the LSTM model.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    _info("🔄 Training started...")

    # set the variable indicating the state of the process
    phase_var.set("training")

    # dataloader setup
    _, training_loader = _loader_setup(
        "training",
        True,
        config_settings
    )

    # setup for training
    device, criterion, model, optimizer = (
        _model_setup(
            config_settings.model_params,
            config_settings.learning_rate,
            _extract_targets_from_loader(training_loader),
            config_settings
        )
    )

    # train the model
    _train_n_epochs(
        config_settings.training_num_epochs,
        model,
        training_loader,
        optimizer,
        criterion,
        device
    )

    # save the trained model
    _save_model(model, config_settings)

    # print a successful message
    _info("✅ Training successfully completed.")