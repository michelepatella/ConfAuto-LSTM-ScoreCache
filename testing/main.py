from utils.AccessLogsDataset import AccessLogsDataset
from utils.log_utils import info, phase_var
from utils.evaluation_utils import evaluate_model
from utils.dataloader_utils import dataloader_setup, extract_targets_from_dataloader
from utils.model_utils import load_model, model_setup


def testing(config_settings):
    """
    Method to test the model.
    :param config_settings: The configuration settings.
    :return: The average loss, the average loss per class,
    the metrics computed, and the cost.
    """
    # initial message
    info("🔄 Testing started...")

    # set the variable indicating the state of the process
    phase_var.set("testing")

    # dataloader setup
    _, testing_loader = dataloader_setup(
        "testing",
        False,
        config_settings,
        AccessLogsDataset
    )

    # setup for testing
    device, criterion, model, _ = (
        model_setup(
            config_settings.model_params,
            config_settings.learning_rate,
            extract_targets_from_dataloader(testing_loader),
            config_settings
        )
    )

    # load the trained model
    model = load_model(
        model,
        device,
        config_settings
    )

    model.eval()

    # evaluate the model
    avg_loss, avg_loss_per_class, metrics, cost_perc, _, _ = evaluate_model(
        model,
        testing_loader,
        criterion,
        device,
        config_settings,
        show_stats=True
    )

    # print a successful message
    info("✅ Testing completed.")

    return avg_loss, avg_loss_per_class, metrics, cost_perc