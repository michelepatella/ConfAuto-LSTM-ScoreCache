from utils.logs.log_utils import info


def calculate_prefetch_hit_rate(
        prefetch_hits,
        tot_prefetch,
):
    """
    Method to compute prefetch hit rate.
    :param prefetch_hits: The number of prefetch hits.
    :param tot_prefetch: Total number of prefetches.
    :return: The prefetch hit rate.
    """
    # initial message
    info("🔄 Prefetch hit rate calculation started...")

    try:
        if tot_prefetch > 0:
            # calculate prefetch hit rate
            prefetch_hit_rate =  prefetch_hits / tot_prefetch
        else:
            prefetch_hit_rate = 0.0
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"ZeroDivisionError: {e}.")
    except NameError as e:
        raise NameError(f"NameError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")


    # show a successful message
    info("🟢 Prefetch hit rate computed.")

    return prefetch_hit_rate