from utils.logs.log_utils import info


def calculate_prefetching_avg_latency(
        autoregressive_latencies
):
    """
    Method to calculate prefetching average latency.
    :return: Prefetching average latency.
    """
    # initial message
    info("🔄 Prefetching average latency calculation started...")

    try:
        avg_prefetching_latency = 0
        if (
            autoregressive_latencies is not None and
            not len(autoregressive_latencies) == 0
        ):
            # filter only not None elements
            valid_latencies = [
                lat for lat in autoregressive_latencies
                if lat is not None
            ]
            if valid_latencies:
                # calculate average prefetching latency
                avg_prefetching_latency = (
                        sum(valid_latencies)/
                        len(valid_latencies)
                )
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(f"ZeroDivisionError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except NameError as e:
        raise NameError(f"NameError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Prefetching average latency calculated.")

    return avg_prefetching_latency