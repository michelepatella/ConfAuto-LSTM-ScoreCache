import os
from utils.logs.log_utils import info, debug


def get_config_abs_path():
    """
    Method to get the absolute path of the config file.
    :return: The absolute path of the config file.
    """
    # initial message
    info("🔄 Config file absolute path started...")

    try:
        # define the absolute path of the config file
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'config.yaml'
        )
        abs_path = os.path.abspath(path)
    except (NameError, TypeError, AttributeError, OSError) as e:
        raise RuntimeError(f"❌ Error while getting the config absolute path: {e}.")

    # debugging
    debug(f"⚙️ Absolute path of config file: {abs_path}.")

    # show a successful message
    info("🟢 Config file absolute path obtained.")

    return abs_path