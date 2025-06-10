import torch
from utils.logs.log_utils import info


def calculate_top_k_accuracy(
        targets,
        outputs,
        config_settings
):
    """
    To calculate the top-k accuracy of the predictions.
    :param targets: The targets.
    :param outputs: The outputs of the model.
    :param config_settings: The configuration settings.
    :return: The top k-accuracy of the predictions.
    """
    # initial message
    info("🔄 Top-k accuracy computation started...")

    try:
        # prepare data
        outputs_tensor = torch.stack(outputs)

        # extract top-k predictions
        top_k_preds = (torch.topk(
            outputs_tensor,
            k=config_settings.top_k,
            dim=1
        ).indices.cpu().numpy())

        # initialize the no. of correct predictions
        correct = 0

        # count the correct predictions
        for i in range(len(targets)):

            # get the top-k predictions
            top_k_i = top_k_preds[i][:config_settings.top_k]

            # check if the target is contained into the
            # top-k predictions
            if targets[i] in top_k_i:
                correct += 1

        # calculate the accuracy
        accuracy = correct / len(targets)
    except RuntimeError as e:
        raise RuntimeError(f"RuntimeError: {e}.")
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"ZeroDivisionError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Top-k accuracy computed.")

    return accuracy