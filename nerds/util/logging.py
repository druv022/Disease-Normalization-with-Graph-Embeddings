import logging
from math import ceil


def get_logger(log_level="DEBUG"):
    # TODO: The log level should be adjusted by some kind of configuration
    # file, e.g. the dev build should have DEBUG, while the release build
    # should have "WARN" or higher.
    f = "%(levelname)s %(asctime)s %(module)s %(filename)s: %(message)s"
    logging.basicConfig(format=f)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    return logger


def log_progress(logger, index, total, message=None):
    if (index + 1) % ceil(total / 100) == 0 or index + 1 == total:
        logger.info("{:.1f}% of documents are processed".format(100 * (index + 1) / total) + (
            " : {}".format(message) if message else ""))
