import numpy as np
from utils.logs.log_utils import info


def generate_access_pattern(
        probs,
        key_range,
        current_time,
        history_keys,
        config_settings
):
    """
    Method to generate access pattern requests.
    :param probs: The probabilities of the keys.
    :param key_range: The range of possible keys.
    :param current_time: The current time.
    :param history_keys: The history keys.
    :param config_settings: The configuration settings.
    :return: The requested generated.
    """
    # initial message
    info("🔄 Access pattern requests generation started...")

    # set data
    hour = (current_time / 3600.0) % 24
    base = config_settings.first_key
    keys = list(range(
        base,
        config_settings.last_key
    ))
    n_keys = len(keys)
    range_size = (
            config_settings.last_key -
            config_settings.first_key
    )
    noise_low, noise_high = config_settings.noise_range

    if len(history_keys) < 5:
        return np.random.choice(keys)

    idx = len(history_keys)

    # 05:00 - 09:00 repetition pattern
    if 5 <= hour < 9:
        if idx % config_settings.repetition_interval == 0:
            new_key = history_keys[-config_settings.repetition_offset]
        else:
            new_key = np.random.choice(
                keys[:n_keys // 3]
            )

    # 09:00 - 12:00 toggle pattern
    elif 9 <= hour < 12:
        toggle = (idx // config_settings.toggle_interval) % 2
        if toggle == 0:
            new_key = ((
                    (history_keys[-1] - base + 1) % range_size)
                    + base
            )
        else:
            new_key = ((
                    (history_keys[-2] - base - 1) % range_size)
                    + base
            )

    # 12:00 - 18:00 cyclic scanning
    elif 12 <= hour < 18:
        cycle_length = (
                config_settings.cycle_base +
                (idx // config_settings.cycle_divisor)
                % config_settings.cycle_mod
        )
        cycle = keys[:cycle_length]
        new_key = cycle[idx % cycle_length]

    # 18:00 - 23:00 distorted history
    elif 18 <= hour < 23:
        if idx % config_settings.distortion_interval == 0:
            new_key = ((
                    (history_keys[-4] - base + 2) % range_size)
                    + base
            )
        else:
            noise = np.random.randint(
                noise_low,
                noise_high + 1
            )
            new_key = ((
                    (history_keys[-1] - base + noise) % range_size)
                    + base
            )

    # 23:00 - 05:00 pattern
    else:
        if idx % config_settings.memory_interval == 0:
            new_key = history_keys[-config_settings.memory_offset]
        else:
            new_key = np.random.choice(
                key_range,
                p=probs
            )

    # show a successful message
    info(f"🟢 Requests access pattern generated.")

    return new_key