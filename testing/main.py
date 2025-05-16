import logging

from main import phase_var
from utils.log_utils import _info
from utils.config_utils import _get_config_value
from utils.evaluation_utils import _evaluate_model
from utils.setup_utils import _training_testing_setup, _loader_setup, _extract_targets_from_loader
from utils.model_utils import _load_model


def testing():
    """
    Method to test the model.
    :return: The average loss, the average loss per class
    and metrics computed.
    """
    # initial message
    _info("🔄 Testing started...")
    phase_var.set("testing")

    # dataloader setup
    _, testing_loader = _loader_setup(
        "testing",
        False
    )

    # setup for testing
    device, criterion, model, _ = (
        _training_testing_setup(
            _get_config_value("model.params"),
            _get_config_value("training.learning_rate"),
            _extract_targets_from_loader(testing_loader)
        )
    )

    # load the trained model
    model = _load_model(
        model,
        _get_config_value("model.model_save_path"),
        device
    )

    model.to(device)

    model.eval()

    # evaluate the model
    avg_loss, avg_loss_per_class, metrics = _evaluate_model(
        model,
        testing_loader,
        criterion,
        device
    )

    # print a successful message
    _info("✅ Testing completed.")

    return avg_loss, avg_loss_per_class, metrics