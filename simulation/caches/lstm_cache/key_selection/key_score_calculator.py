from utils.logs.log_utils import info


def calculate_key_scores(
        num_keys,
        num_steps,
        prob_matrix,
        conf_matrix
):
    """
    Method to calculate key scores.
    :param num_keys: The number of keys.
    :param num_steps: The number of steps.
    :param prob_matrix: The probability matrix.
    :param conf_matrix: The confidence matrix.
    :return: The key scores.
    """
    # initial message
    info("🔄 Key scores calculation started...")

    try:
        scores = {}

        # for each key calculate a score
        for k in range(num_keys):
            score = 0.0
            for t in range(num_steps):
                # calculate the final score as a combination of
                # probability of a key of being used and CIs related
                # to that prediction
                score += prob_matrix[t, k] * (conf_matrix[t, k] + 0.5)
            scores[k] = score

        # normalize scores in [0,1]
        min_score = min(scores.values())
        max_score = max(scores.values())
        score_range = max_score - min_score \
            if max_score != min_score \
            else 1.0
        scores = {
            k: (v - min_score) / score_range
            for k, v in scores.items()
        }
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"ZeroDivisionError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # print a successful message
    info("🟢 Key scores calculated.")

    return scores