import contextvars
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(phase)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("log.log"),
        logging.StreamHandler()
    ]
)

def _info(msg, *args, **kwargs):
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


def _debug(msg, *args, **kwargs):
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

# contextual variable indicating the phase
# in which the logging message is located in
phase_var = contextvars.ContextVar("phase", default="unknown")