import logging
from main import phase_var


def _info(msg, *args, **kwargs):
    """
    Method to print a logging info message.
    :param msg: The message to print.
    :param args: The arguments to print.
    :param kwargs: The keyword arguments to print.
    :return:
    """
    logging.info(
        msg,
        *args,
        extra={"phase": phase_var.get()},
        **kwargs
    )


def _debug(msg, *args, **kwargs):
    """
    Method to print a logging debug message.
    :param msg: The message to print.
    :param args: The arguments to print.
    :param kwargs: The keyword arguments to print.
    :return:
    """
    logging.debug(
        msg,
        *args,
        extra={"phase": phase_var.get()},
        **kwargs
    )