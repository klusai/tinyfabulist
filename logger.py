import logging

LOG_FILE = 'tinyfabulist.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

class TinyFabulistError(Exception):
    pass

class ConfigError(TinyFabulistError):
    pass

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE)
        ]
    )
    return logging.getLogger(__name__)
