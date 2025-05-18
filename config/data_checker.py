def _check_distribution(
        seed,
        distribution_type,
        num_requests,
        num_keys,
        first_key,
        last_key,
        freq_windows
):
    """
    Method to check data distribution parameters.
    :param seed: The random seed.
    :param distribution_type: The distribution type.
    :param num_requests: The number of requests.
    :param num_keys: The number of keys.
    :param first_key: The first key.
    :param last_key: The last key.
    :param freq_windows: The frequency windows.
    :return:
    """
    # check seed
    if seed < 0 or not isinstance(seed, int):
        raise RuntimeError("❌ 'data.distribution.seed' must be an integer >= 0.")

    # check distribution type
    if distribution_type not in {"static", "dynamic"}:
        raise RuntimeError("❌ 'data.distribution.type' must be 'static' or 'dynamic'.")

    # check number of requests
    if (
        num_requests <= 0
        or not isinstance(num_requests, int)
    ):
        raise RuntimeError("❌ 'data.distribution.num_requests' must be an integer > 0.")

    # check number of keys
    if not isinstance(num_keys, int) or num_keys <= 1:
        raise RuntimeError("❌ 'data.distribution.num_keys' must be an integer > 1.")
    if abs(last_key - first_key) != num_keys:
        raise RuntimeError("❌ 'data.distribution.num_keys' must match the"
                           " range defined by first_key and last_key.")

    # check first key and last key
    if (
        not isinstance(first_key, int) or
        not isinstance(last_key, int)
        or last_key <= first_key
    ):
        raise RuntimeError("❌ 'data.distribution.key_range.first_key' and  "
                           "'data.distribution.key_range.last_key' "
                           "must be integers, with the first one "
                           "< the second one.")

    # check frequency windows
    if (
        not isinstance(freq_windows, list)
        or not freq_windows
        or not all(isinstance(w, int) and w > 0 for w in freq_windows)
    ):
        raise RuntimeError("❌ 'data.distribution.freq_windows' must "
                           "be a list of integers > 0.")


def _check_access_pattern(
            zipf_alpha,
            zipf_alpha_start,
            zipf_alpha_end,
            zipf_time_steps,
            locality_prob,
            burst_high,
            burst_low,
            burst_every,
            burst_peak,
            periodic_base_scale,
            periodic_amplitude
    ):
    """
    Method to check data access pattern parameters.
    :param zipf_alpha: Zipf alpha value.
    :param zipf_alpha_start: Zipf alpha start value.
    :param zipf_alpha_end: Zipf alpha end value.
    :param zipf_time_steps: Zipf time steps value.
    :param locality_prob: Locality probability value.
    :param burst_high: The burst high value.
    :param burst_low: The burst low value.
    :param burst_every: The burst every value.
    :param burst_peak: The burst peak value.
    :param periodic_base_scale: The periodic base scale value.
    :param periodic_amplitude: The periodic amplitude value.
    :return:
    """
    # check zipf alpha
    if (
        not isinstance(zipf_alpha, float)
        or zipf_alpha <= 0
    ):
        raise RuntimeError("❌ 'data.access_pattern.zipf.alpha' must be a float > 0.")

    # check zipf alpha start and end
    if(
        not all(isinstance(v, float)
        and v > 0 for v in [zipf_alpha_start, zipf_alpha_end])
    ):
        raise RuntimeError("❌ 'data.access_pattern.zipf.alpha_start' and "
                           "'data.access_pattern.zipf.alpha_end' must be float > 0.")
    if zipf_alpha_end < zipf_alpha_start:
        raise RuntimeError("❌ 'data.access_pattern.zipf.alpha_end' must be"
                           " >= 'data.access_pattern.zipf.alpha_start'.")

    # check time steps
    if (
        not isinstance(zipf_time_steps, int)
        or zipf_time_steps <= 0
    ):
        raise RuntimeError("❌ 'data.access_pattern.zipf.time_steps"
                           "' must be an integer > 0.")

    # check local probability
    if (
        not isinstance(locality_prob, float)
        or not (0.0 <= locality_prob <= 1.0)
    ):
        raise RuntimeError("❌ 'data.access_pattern.locality.prob' must"
                           " be a float between 0.0 and 1.0.")

    # check burst high and low
    for name, val in [("burst_high", burst_high), ("burst_low", burst_low)]:
        if not isinstance(val, float) or val <= 0:
            raise RuntimeError(f"'data.temporal_pattern.burstiness.{name}' must be a float > 0.")
    if burst_high >= burst_low:
        raise RuntimeError("❌ 'data.temporal_pattern.burstiness.burst_high'"
                           " must be < 'data.temporal_pattern.burstiness.burst_low'.")

    # check burst every
    if (
        not isinstance(burst_every, int)
        or burst_every <= 0
    ):
        raise RuntimeError("❌ 'data.temporal_pattern.burstiness.burst_every'"
                           " must be an integer > 0.")

    # check burst peak
    if (
        not isinstance(burst_peak, int)
        or burst_peak < 0
    ):
        raise RuntimeError("❌ 'data.temporal_pattern.burstiness.burst_peak'"
                           " must be an integer >= 0.")

    # check periodic base scale
    if (
        not isinstance(periodic_base_scale, (int, float))
        or periodic_base_scale <= 0
    ):
        raise RuntimeError("❌ 'data.temporal_pattern.periodic.base_scale'"
                           " must be a number > 0.")

    # check periodic amplitude
    if (
        not isinstance(periodic_amplitude, (int, float))
        or periodic_amplitude < 0
    ):
        raise RuntimeError("❌ 'data.temporal_pattern.periodic.periodic_amplitude'"
                           " must be a number >= 0.")


def _check_sequence(
        seq_len,
        embedding_dim,
        num_requests
):
    """
    Method to check sequence parameters.
    :param seq_len: The sequence length.
    :param embedding_dim: The embedding dimension.
    :param num_requests: The number of requests.
    :return:
    """
    # check sequence length
    if (
        not isinstance(seq_len, int)
        or seq_len <= 0
    ):
        raise RuntimeError("❌ 'data.sequence.len' must be an integer > 0.")
    if seq_len > num_requests:
        raise RuntimeError("❌ 'data.sequence.len' must be"
                           " <= 'data.distribution.num_requests'.")

    # check embedding dimension
    if (
        not isinstance(embedding_dim, int)
        or embedding_dim <= 0
    ):
        raise RuntimeError("❌ 'data.sequence.embedding_dim'"
                           " must be an integer > 0.")


def _check_dataset(training_perc, static_save_path, dynamic_save_path):
    """
    Method to check dataset parameters.
    :param training_perc: The training set percentage.
    :param static_save_path: The path to save the static dataset.
    :param dynamic_save_path: The path to save the dynamic dataset.
    :return:
    """
    # check training percentage
    if (
        not isinstance(training_perc, float)
        or not (0.0 <= training_perc <= 1.0)
    ):
        raise RuntimeError("❌ 'data.dataset.training_perc'"
                           " must be a float between 0.0 and 1.0.")

    # check dataset paths
    if(
        not isinstance(static_save_path, str)
        or not isinstance(dynamic_save_path, str)
    ):
        raise RuntimeError("❌ Both 'data.dataset.static_save_path' and "
                           "'data.dataset.dynamic_save_path' must be strings.")