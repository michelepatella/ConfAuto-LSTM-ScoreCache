from utils.data.AccessLogsDataset import AccessLogsDataset
from utils.data.dataloader.dataloader_setup import dataloader_setup
from utils.logs.log_utils import info
from utils.model.setup.trained_model_setup import trained_model_setup


def simulation_setup(
        policy_name,
        config_settings
):
    """
    Method to set up the simulation environment.
    :param policy_name: The policy name.
    :param config_settings: The configuration settings.
    :return: All the data needed to run the simulation.
    """
    # initial message
    info("🔄 Simulation setup started...")

    # initialize data
    (
        device,
        criterion,
        model
    ) = None, None, None
    counters = {
        'hits': 0,
        'misses': 0,
        'hits_cold_start': 0
    }
    timeline = []
    recent_hits = []
    prefetching_latency = []
    window = config_settings.prediction_interval

    # get the testing set
    (
        testing_set,
        testing_loader
    ) = dataloader_setup(
        "testing",
        config_settings.testing_batch_size,
        False,
        config_settings,
        AccessLogsDataset
    )

    # initial model setup, in case of LSTM cache
    if policy_name == 'LSTM':
        # setup for lstm cache
        (
            device,
            criterion,
            model
        ) = trained_model_setup(
            testing_loader,
            config_settings
        )

        try:
            model.eval()
            model.to(device)
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

    # print a successful message
    info("🟢 Simulation setup completed.")

    return (
        counters,
        timeline,
        recent_hits,
        prefetching_latency,
        window,
        testing_set,
        testing_loader,
        device,
        criterion,
        model,
    )