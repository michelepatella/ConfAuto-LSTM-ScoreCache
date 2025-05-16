from collections import Counter
from utils.config_utils import _get_config_value
from utils.log_utils import _info, _debug


def _generate_last_freq(sequence):
    """
    Method to generate the last frequencies of a given sequence.
    :param sequence: The sequence to generate frequencies for.
    :return: The generated frequencies.
    """
    # read configuration
    windows = _get_config_value("data.freq_windows")

    try:
        # freq columns dictionary initialization
        freq_columns = {}

        if sequence is not None:
            # create new column
            for w in windows:
                col_name = f"freq_last_{w}"
                freq_columns[col_name] = _compute_frequency(
                    sequence,
                    window=w
                )
    except Exception as e:
        raise Exception(f"❌ Error while generating features: {e}")

    return freq_columns


def _compute_frequency(sequence, window):
    """
    Method to compute the frequency of a specific sequence in a given window.
    :param sequence: The sequence to compute the frequency of.
    :param window: The window within to compute the frequency.
    :return: The frequency of the sequence.
    """
    # initial message
    _info("🔄 Frequency sequence counting started...")

    # debugging
    _debug(f"⚙️ Sequence for which to count the frequency: {sequence}.")
    _debug(f"⚙️ Window: {window}.")

    try:
        # initialize frequency
        freq = []

        # count the frequency of the sequence
        # within the given temporal window
        for i in range(len(sequence)):
            if i < window:
                recent = sequence[:i]
            else:
                recent = sequence[i - window:i]
            count = Counter(recent)
            freq.append(count[sequence[i]])

    except Exception as e:
        raise Exception(f"❌ Error while computing the frequency of sequence: {e}")

    # debugging
    _debug(f"⚙️ Frequency computed (sequence-frequency): ({sequence} - {freq}).")

    # show a successful message
    _info(f"🟢 Frequency of the sequence counted.")

    return freq
