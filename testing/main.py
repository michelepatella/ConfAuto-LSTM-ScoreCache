from utils.log_utils import _info, phase_var
from utils.evaluation_utils import _evaluate_model
from utils.data_utils import _loader_setup, _extract_targets_from_loader
from utils.model_utils import _load_model, _model_setup


def testing():
    """
    Method to test the model.
    :return: The average loss, the average loss per class
    and metrics computed.
    """
    from main import config_settings

    # initial message
    _info("🔄 Testing started...")

    # set the variable indicating the state of the process
    phase_var.set("testing")

    # dataloader setup
    _, testing_loader = _loader_setup(
        "testing",
        False,
        config_settings.config
    )

    # setup for testing
    device, criterion, model, _ = (
        _model_setup(
            config_settings.model_params,
            config_settings.learning_rate,
            _extract_targets_from_loader(testing_loader)
        )
    )

    # load the trained model
    model = _load_model(
        model,
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