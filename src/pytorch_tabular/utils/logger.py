import logging
import os

from rich.logging import RichHandler


def get_logger(name):
    logger = logging.getLogger(name)
    # ch = logging.StreamHandler()
    logger.setLevel(level=os.environ.get("PT_LOGLEVEL", "INFO"))
    formatter = logging.Formatter("%(asctime)s - {%(name)s:%(lineno)d} - %(levelname)s - %(message)s")
    if not logger.hasHandlers():
        ch = RichHandler(show_level=False, show_time=False, show_path=False, rich_tracebacks=True)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False
    return logger
