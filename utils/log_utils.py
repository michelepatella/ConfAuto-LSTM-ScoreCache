import contextvars
import logging
from logging.handlers import RotatingFileHandler

# contextual variable indicating the phase
# in which the logging message is located in
phase_var = contextvars.ContextVar(
    "phase",
    default="unknown"
)

# define a global formatter
formatter = logging.Formatter('[%(phase)s] %(levelname)s: %(message)s')

# all logging messages from INFO-level must be written in a file
file_handler = RotatingFileHandler(
    './logs/logs.log',
    maxBytes=10_000_000,
    backupCount=100
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# show in terminal only ERROR-level logging messages
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

# configuration of logging messages
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, stream_handler]
)


def info(msg, *args, **kwargs):
    """
    Method to print a logging info message.
    :param msg: The message to print.
    :param args: The arguments to print.
    :param kwargs: The keyword arguments to print.
    :return:
    """
    try:
        logging.info(
            msg,
            *args,
            extra={"phase": phase_var.get()},
            **kwargs
        )
    except (KeyError, ValueError, LookupError, TypeError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while logging info message: {e}.")


def debug(msg, *args, **kwargs):
    """
    Method to print a logging debug message.
    :param msg: The message to print.
    :param args: The arguments to print.
    :param kwargs: The keyword arguments to print.
    :return:
    """
    try:
        logging.debug(
            msg,
            *args,
            extra={"phase": phase_var.get()},
            **kwargs
        )
    except (KeyError, ValueError, LookupError, TypeError, AttributeError) as e:
        raise RuntimeError(f"❌ Error while logging debug message: {e}.")