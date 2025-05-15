import logging
from utils.config_utils import _get_config_value
from utils.evaluation_utils import _evaluate_model
from utils.setup_utils import _training_testing_setup, _loader_setup, _extract_targets_from_loader, _load_trained_model


def testing():
    """
    Method to test the model.
    :return: The average loss and metrics computed.
    """
    # initial message
    logging.info("🔄 Testing started...")

    # dataloader setup
    _, testing_loader = _loader_setup("testing", False)

    # setup for testing
    device, criterion, model, _ = (
        _training_testing_setup(
            _get_config_value("model.params"),
            _get_config_value("training.learning_rate"),
            _extract_targets_from_loader(testing_loader)
        )
    )

    # load the pretrained model
    model = _load_trained_model(
        model,
        _get_config_value("model.model_save_path"),
        device
    )

    model.to(device)

    model.eval()

    # evaluate the model (by only average loss, no metrics)
    avg_loss, avg_loss_per_class, metrics = _evaluate_model(
        model,
        testing_loader,
        criterion,
        device
    )

    # print a successful message
    logging.info("✅ Testing completed.")

    return avg_loss, avg_loss_per_class, metrics