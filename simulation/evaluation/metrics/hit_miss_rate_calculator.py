from utils.logs.log_utils import info


def calculate_hit_miss_rate(counters):
    """
    Method to calculate hit and miss rate.
    :param counters: A counter used while simulating a cache policy
    :return: The hit and miss rate in terms of %.
    """
    # initial message
    info("🔄 Hit and miss rate calculation started...")

    try:
        # calculate hit rate and miss rate in terms of %
        total = counters['hits'] + counters['misses']
        hit_rate = counters['hits'] / total * 100
        miss_rate = counters['misses'] / total * 100
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"ZeroDivisionError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Hit and miss rate calculated.")

    return hit_rate, miss_rate