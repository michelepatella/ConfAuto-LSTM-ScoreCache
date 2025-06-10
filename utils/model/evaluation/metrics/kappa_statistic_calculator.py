from sklearn.metrics import cohen_kappa_score
from utils.logs.log_utils import info


def calculate_kappa_statistic(
        targets,
        predictions
):
    """
    Method to calculate the kappa statistic.
    :param targets: The targets.
    :param predictions: The predictions coming from the model.
    :return: The kappa statistic.
    """
    # initial message
    info("🔄 Kappa statistic calculation started...")

    try:
        # calculate kappa statistic
        kappa = cohen_kappa_score(
            targets,
            predictions
        )
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Kappa statistic calculated.")

    return kappa