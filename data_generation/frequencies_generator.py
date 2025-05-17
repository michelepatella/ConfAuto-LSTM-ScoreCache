from collections import Counter
from config.main import freq_windows
from utils.log_utils import _info, _debug


def _calculate_rel_frequency(sequence, window):
    """
    Method to calculate the relative frequency of a specific sequence in a given window.
    :param sequence: The sequence to calculate the frequency of.
    :param window: The window within to calculate the frequency.
    :return: The frequency of the sequence.
    """
    # initial message
    _info("🔄 Relative frequency sequence calculation started...")

    # debugging
    _debug(f"⚙️ Sequence length: {len(sequence)}.")
    _debug(f"⚙️ Window: {window}.")

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
    _debug(f"⚙️ Frequencies length: {len(freqs)}.")

    # show a successful message
    _info(f"🟢 Relative frequency of the sequence calculated.")

    return freqs


def _generate_last_freq(sequence):
    """
    Method to generate the last frequencies of a given sequence.
    :param sequence: The sequence to generate frequencies for.
    :return: The generated frequencies.
    """
    # initial message
    _info("🔄 Frequencies generation started...")

    try:
        # freq columns dictionary initialization
        freq_columns = {}

        # create a new column
        for w in freq_windows:
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