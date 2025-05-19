from collections import Counter
from utils.log_utils import info, debug


def _calculate_rel_frequency(sequence, window):
    """
    Method to calculate the relative frequency of a specific sequence in a given window.
    :param sequence: The sequence to calculate the frequency of.
    :param window: The window within to calculate the frequency.
    :return: The frequency of the sequence.
    """
    # initial message
    info("🔄 Relative frequency sequence calculation started...")

    # debugging
    debug(f"⚙️ Sequence length: {len(sequence)}.")
    debug(f"⚙️ Window: {window}.")

    # initialize frequencies
    freqs = []

    try:
        # count the frequency of the sequence
        # within the given temporal window
        for i in range(len(sequence)):
            if i < window:
                recent = sequence[:i]
            else:
                recent = sequence[i - window:i]

            # calculate the relative frequency
            count = Counter(recent)
            freq = count[sequence[i]] / len(recent) \
                if len(recent) > 0 \
                else 0.0
            freqs.append(freq)
    except (TypeError, ZeroDivisionError,
            IndexError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while calculating relative frequency sequence: {e}.")

    # debugging
    debug(f"⚙️ Frequencies length: {len(freqs)}.")

    # show a successful message
    info(f"🟢 Relative frequency of the sequence calculated.")

    return freqs


def _generate_last_rel_freq(sequence, config_settings):
    """
    Method to generate the last relative frequencies of a given sequence.
    :param sequence: The sequence to generate frequencies for.
    :param config_settings: The configuration settings.
    :return: The generated frequencies.
    """
    # initial message
    info("🔄 Relative frequencies generation started...")

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
        debug(f"⚙️ Relative frequency columns length: {len(freq_columns)}.")

        # show a successful message
        info("🟢 Relative frequencies generated.")

        return freq_columns

    except (ValueError, TypeError, NameError, RuntimeError) as e:
        raise RuntimeError(f"❌ Error while generating relative frequencies: {e}.")