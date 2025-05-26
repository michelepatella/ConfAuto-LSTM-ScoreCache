from utils.AccessLogsDataset import AccessLogsDataset
from utils.log_utils import info, phase_var
from utils.evaluation_utils import evaluate_model
from utils.dataloader_utils import dataloader_setup, extract_targets_from_dataloader
from utils.model_utils import load_model, model_setup


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
    device, criterion, model, _ = (
        model_setup(
            config_settings.model_params,
            config_settings.learning_rate,
            extract_targets_from_dataloader(testing_loader),
            config_settings
        )
    )

    from collections import Counter

    # Estrai tutte le etichette dal testing_loader
    all_targets = []
    for _, targets in testing_loader:
        if isinstance(targets, dict):
            targets = targets["request"]  # usa la chiave corretta se è un dizionario
        all_targets.extend(targets.cpu().numpy())  # assicurati che sia una lista di numeri

    # Calcola la frequenza per ogni classe
    class_counts = Counter(all_targets)

    # Stampa le frequenze
    info("📊 Frequenze delle classi nel testing set:")
    for cls, count in sorted(class_counts.items()):
        info(f"Classe {cls}: {count} campioni")

    # load the trained model
    model = load_model(
        model,
        device,
        config_settings
    )

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