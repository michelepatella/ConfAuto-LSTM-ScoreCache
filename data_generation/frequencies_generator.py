from utils.log_utils import _info, _debug
from utils.metrics_utils import _calculate_rel_frequency


def _generate_last_freq(sequence, config_settings):
    """
    Method to generate the last frequencies of a given sequence.
    :param sequence: The sequence to generate frequencies for.
    :param config_settings: The configuration settings.
    :return: The generated frequencies.
    """
    # initial message
    _info("🔄 Frequencies generation started...")

    try:
        # freq columns dictionary initialization
        freq_columns = {}

        # create a new column
        for w in config_settings.freq_windows:
            col_name = f"freq_last_{w}"
            freq_columns[col_name] = _calculate_rel_frequency(
                sequence,
                window=w
            )

        # debugging
        _debug(f"⚙️ Frequency columns length: {len(freq_columns)}.")

        # show a successful message
        _info("🟢 Frequencies generated.")

        return freq_columns

    except (ValueError, TypeError, NameError, RuntimeError) as e:
        raise RuntimeError(f"❌ Error while generating frequencies: {e}.")