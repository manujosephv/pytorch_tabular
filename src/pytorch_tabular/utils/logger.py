import logging
import os

from rich.logging import RichHandler


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level=os.environ.get("PT_LOGLEVEL", "INFO"))
    formatter = logging.Formatter("%(asctime)s - {%(name)s:%(lineno)d} - %(levelname)s - %(message)s")
    # ch = logging.StreamHandler()
    ch = RichHandler(show_level=False, show_time=False, show_path=False, rich_tracebacks=True)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
