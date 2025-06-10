import random
from utils.logs.log_utils import info


def manage_cold_start(
        cache,
        current_time,
        config_settings
):
    """
    Method to handle cache cold start.
    :param cache: The cache for which to handle cold start.
    :param current_time: The current time.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Cold start handling started...")

    try:
        # select random keys
        all_possible_keys = list(range(
            config_settings.num_keys
        ))
        random_keys = random.sample(
            all_possible_keys,
            min(
                config_settings.cache_size,
                len(all_possible_keys)
            )
        )

        # fill the cache with these keys having 0 score
        for k in random_keys:
            cache.put(
                k,
                0.0,
                current_time,
                cold_start=True,
                config_settings=config_settings
            )
    except NameError as e:
        raise NameError(f"NameError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except KeyError as e:
        raise KeyError(f"KeyError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # print a successful message
    info("🟢 Cold start handled.")