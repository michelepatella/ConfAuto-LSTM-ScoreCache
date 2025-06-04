from utils.logs.log_utils import info


def trace_hits_misses(
        counters,
        prev_hits,
        recent_hits,
        window,
        idx,
        timeline
):
    """
    Method to trace hits and misses over time.
    :param counters: A counter used while simulating the cache policy.
    :param prev_hits: The previous number of hits.
    :param recent_hits: The recent number of hits.
    :param window: A window in which to calculate hits and misses.
    :param idx: The index of the current request.
    :param timeline: The timeline of hits and misses.
    :return: The number of recent hits and timeline updated.
    """
    # initial message
    info("🔄 Number of hits and misses counting started...")

    try:
        # keep track of the number of hits each time
        was_hit = counters['hits'] > prev_hits
        recent_hits.append(1 if was_hit else 0)
        if len(recent_hits) > window:
            recent_hits.pop(0)

        # calculate instant hit rate and overall average hit rate
        instant_hit_rate = counters['hits'] / (idx + 1)

        # store timeline data
        timeline.append({
            'index': idx,
            'instant_hit_rate': instant_hit_rate,
            'total_hits': counters['hits'],
            'total_misses': counters['misses']
        })
    except (
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
            ZeroDivisionError
    ) as e:
        raise RuntimeError(f"❌ Error while keeping track of the no. of hits and misses: {e}.")

    # print a successful message
    info("🟢 Number of hits and misses counted.")

    return recent_hits, timeline