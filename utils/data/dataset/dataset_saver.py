from utils.data.dataset.dataset_utils import get_dataset_path_type
from utils.logs.log_utils import info, debug


def save_dataset(
        df,
        config_settings
):
    """
    Method to save the dataset.
    :param df: Dataframe to save.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Dataset saving started...")

    # get the dataset path
    dataset_path = get_dataset_path_type(config_settings)

    # debugging
    debug(f"⚙️ Dataset shape to save: {df.shape}.")
    debug(f"⚙️ Dataset path: {dataset_path}.")

    try:
        # convert dataframe to CSV file
        df.to_csv(
            dataset_path,
            index=False
        )
    except OSError as e:
        raise OSError(f"OSError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info(f"🟢 Dataset saved to '{dataset_path}'.")
