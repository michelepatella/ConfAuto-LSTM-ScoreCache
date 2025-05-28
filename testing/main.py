from utils.AccessLogsDataset import AccessLogsDataset
from utils.log_utils import info, phase_var
from utils.evaluation_utils import evaluate_model
from utils.dataloader_utils import dataloader_setup
from utils.model_utils import trained_model_setup


def testing(config_settings):
    """
    Method to test the model.
    :param config_settings: The configuration settings.
    :return: The average loss and the metrics computed.
    """
    # initial message
    info("🔄 Testing started...")

    # set the variable indicating the state of the process
    phase_var.set("testing")

    # dataloader setup
    _, testing_loader = dataloader_setup(
        "testing",
        config_settings.testing_batch_size,
        False,
        config_settings,
        AccessLogsDataset
    )

    # setup for testing
    (
        device,
        criterion,
        model
    ) = trained_model_setup(testing_loader, config_settings)

    model.eval()

    # evaluate the model
    avg_loss, metrics, *_ = evaluate_model(
        model,
        testing_loader,
        criterion,
        device,
        config_settings,
        compute_metrics=True
    )

    # print a successful message
    info("✅ Testing completed.")

    return avg_loss, metrics