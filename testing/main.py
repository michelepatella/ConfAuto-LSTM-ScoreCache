import torch
from testing.visualization.testing_plotter import plot_precision_recall_curve, plot_confusion_matrix
from utils.data.AccessLogsDataset import AccessLogsDataset
from utils.logs.log_utils import info, phase_var
from utils.data.dataloader.dataloader_setup import dataloader_setup
from utils.model.evaluation.model_evaluator import evaluate_model
from utils.model.setup.trained_model_setup import trained_model_setup
from testing.visualization.testing_reporter import generate_model_evaluation_report


def testing(config_settings):
    """
    Method to test the model.
    :param config_settings: The configuration settings.
    :return:
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
    ) = trained_model_setup(
        testing_loader,
        config_settings
    )

    model.eval()

    # evaluate the model
    (
        avg_loss,
        metrics,
        all_outputs,
        all_targets, _) = evaluate_model(
        model,
        testing_loader,
        criterion,
        device,
        config_settings,
        compute_metrics=True
    )

    try:
        # show results
        generate_model_evaluation_report(
            metrics["class_report"],
            metrics["top_k_accuracy"],
            metrics["kappa_statistic"],
            avg_loss,
            config_settings
        )

        # show some plots
        plot_precision_recall_curve(
            all_targets,
            torch.stack(all_outputs).numpy(),
            config_settings.num_keys
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"]
        )
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except NameError as e:
        raise NameError(f"NameError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # print a successful message
    info("✅ Testing completed.")

