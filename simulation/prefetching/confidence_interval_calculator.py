import torch
from scipy.stats import norm
from utils.logs.log_utils import info, debug


def calculate_confidence_interval(
        all_outputs,
        all_vars,
        config_settings
):
    """
    Method to calculate confidence intervals.
    :param all_outputs: The outputs of the model.
    :param all_vars: The variances of the outputs.
    :param config_settings: Configuration settings.
    :return: The upper and lower confidence interval boundaries.
    """
    # initial message
    info("🔄 Confidence intervals calculation started...")

    try:
        # calculate the z-score
        z_score = norm.ppf(
            1 - (1 - config_settings.confidence_level) / 2
        )

        # debugging
        debug(f"⚙️ Z-score for CIs: {z_score}.")

        # calculate the standard deviation
        outputs_tensor = torch.stack(all_outputs)
        outputs_std = torch.sqrt(
            torch.stack(all_vars)
        )

        # calculate lower and upper CIs boundaries
        lower_ci = (
                outputs_tensor -
                z_score * outputs_std
        )
        upper_ci = (
            outputs_tensor +
            z_score * outputs_std
        )
    except NameError as e:
        raise NameError(f"NameError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Confidence intervals calculated.")

    return lower_ci, upper_ci