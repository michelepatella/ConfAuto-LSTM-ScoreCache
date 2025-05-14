import logging
from utils.config_utils import _get_config_value
from utils.evaluation_utils import _evaluate_model
from utils.setup_utils import _training_testing_setup, _loader_setup


def testing():
    """
    Method to test the model.
    :return: The average loss and metrics computed.
    """
    # initial message
    logging.info("🔄 Testing started...")

    # dataloader setup
    testing_loader = _loader_setup("testing")

    # setup for testing
    device, criterion, model, optimizer = (
        _training_testing_setup(
            _get_config_value("model.params"),
            _get_config_value("training.learning_rate")
        )
    )

    model.eval()

    # evaluate the model (by only average loss, no metrics)
    avg_loss, metrics = _evaluate_model(
        model,
        testing_loader,
        criterion,
        device
    )

    # print a successful message
    logging.info("✅ Testing completed.")

    return avg_loss, metrics