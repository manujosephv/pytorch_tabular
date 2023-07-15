import logging
import os


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level=os.environ.get("PT_LOGLEVEL", "INFO"))
    formatter = logging.Formatter(
        "%(asctime)s - {%(name)s:%(lineno)d} - %(levelname)s - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
