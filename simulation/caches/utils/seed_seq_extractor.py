from utils.data.AccessLogsDataset import AccessLogsDataset
from utils.logs.log_utils import info


def extract_seed_seq(
        current_idx,
        testing_set,
        config_settings
):
    """
    Method to extract seed sequence used by
     autoregressive rollout.
    :param current_idx: The current request index.
    :param testing_set: The testing set.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Seed sequence extraction started...")

    try:
        # define a mobile window sliding over the testing set
        start_idx = current_idx - config_settings.seq_len + 1
        end_idx = current_idx + 1
        testing_window_df = testing_set.data.iloc[start_idx:end_idx + 1]

        # check if enough data contained in the sliding window
        if len(testing_window_df) < config_settings.seq_len:
            return

        # create a dataset from the sliding window
        testing_window_dataset = AccessLogsDataset.from_dataframe(
            testing_window_df,
            config_settings
        )

        # check if there is at least one element in the
        # dataset created
        if len(testing_window_dataset) == 0:
            return

        # extract seed from the testing window
        seed_seq = testing_window_dataset.__getitem__(
            len(testing_window_dataset) - 1
        )
    except NameError as e:
        raise NameError(f"NameError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")


    # print a successful message
    info("🟢 Seed sequence extracted.")

    return seed_seq