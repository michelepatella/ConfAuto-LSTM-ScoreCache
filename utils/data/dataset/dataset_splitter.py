from torch.utils.data import Subset
from utils.logs.log_utils import info


def split_training_set(
        training_set,
        config_settings,
        training_indices=None,
        validation_indices=None
):
    """
    Method to split the dataset into training and validation sets.
    :param training_set: The training set to split.
    :param config_settings: The configuration settings.
    :param training_indices: Validation set index.
    :param validation_indices: Training set index.
    :return: The training and validation sets.
    """
    # initial message
    info("🔄 Training splitting started...")

    try:
        if (
            training_indices is None or
            validation_indices is None
        ):
            # calculate total training set size
            total_training_size = len(training_set)

            # calculate training and validation size
            training_size = int(
                (1.0 - config_settings.validation_perc) *
                total_training_size
            )
            validation_size = int(
                config_settings.validation_perc *
                total_training_size
            )

            # create indexes for training and validation
            training_indices = list(range(
                0,
                training_size
            ))
            validation_indices = list(range(
                training_size,
                training_size + validation_size
            ))

        # split the training set into training and validation set
        final_training_set = Subset(
            training_set,
            training_indices
        )
        final_validation_set = Subset(
            training_set,
            validation_indices
        )
    except (
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
        NameError
    ) as e:
        raise RuntimeError(f"❌ Error while splitting training set into training and validation sets: {e}.")

    # show a successful message
    info("🟢 Training set split.")

    return (
        final_training_set,
        final_validation_set
    )