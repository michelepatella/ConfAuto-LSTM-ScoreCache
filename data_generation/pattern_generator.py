import numpy as np
from config.main import first_key, last_key, periodic_amplitude, burst_every, burst_peak, \
    burst_high, burst_low, periodic_base_scale
from utils.log_utils import _debug, _info


def _generate_pattern(probs, num_requests, timestamps):
    """
    Generates requests and delta times based on a combination
    of bursty and periodic pattern.
    :param probs: The Zipf probabilities.
    :param num_requests: The number of requests.
    :param timestamps: The list of timestamps.
    :return: A tuple containing the generated requests and delta_times.
    """
    # initial message
    _info("🔄 Pattern generation started...")

    # debugging
    _debug(f"⚙️ Probabilities length: {len(probs)}.")
    _debug(f"⚙️ Number of requests: {num_requests}.")
    _debug(f"⚙️ Timestamps length: {len(timestamps)}.")

    # initialize data
    requests = []
    delta_times = []

    # define the day as period (24(h) * 60 (min) * 60(s))
    period = 24 * 60 * 60

    # debugging
    _debug(f"⚙️ Period: {period}.")

    # check timestamps and probabilities
    if len(timestamps) == 0:
        raise ValueError("❌ timestamps list cannot be empty.")

    try:
        for i in range(num_requests):

            # generate the request
            request = np.random.choice(
                np.arange(first_key, last_key),
                p=probs
            )

            # calculate periodic component for frequency scaling
            periodic_scale = (periodic_base_scale +
                              periodic_amplitude *
                              np.cos(2 * np.pi * timestamps[-1] / period))

            # introduction of burstiness
            if i % burst_every < burst_peak:
                # high frequency
                bursty_scale = burst_high
            else:
                # low frequency
                bursty_scale = burst_low

            # combine periodic and bursty scales
            freq_scale = periodic_scale * bursty_scale

            # calculate delta time
            delta_t = np.random.exponential(scale=freq_scale)

            # store generated data
            requests.append(request)
            timestamps.append(timestamps[-1] + delta_t)
            delta_times.append(delta_t)

            # debugging
            _debug(f"⚙️ Number of request generated: {i}.")
            _debug(f"⚙️ Frequency scale calculated: {freq_scale}.")
            _debug(f"⚙️ Request generated: {request}.")
            _debug(f"⚙️ Delta time generated: {delta_t}.")

    except (ValueError, TypeError, IndexError, ZeroDivisionError,
            AttributeError, MemoryError) as e:
        raise RuntimeError(f"❌ Error while generating data access pattern: {e}.")

    # show a successful message
    _info(f"🟢 Pattern generated.")

    return requests, delta_times