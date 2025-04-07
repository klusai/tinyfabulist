import logging
import sys

LOG_FILE = "tinyfabulist.log"
LOG_FORMAT = "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO


class TinyFabulistError(Exception):
    pass


class ConfigError(TinyFabulistError):
    pass


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding='utf-8')
        ],
    )
    return logging.getLogger(__name__)
