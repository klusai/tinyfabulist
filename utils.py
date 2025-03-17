import yaml

from tiny_fabulist.logger import *

# Constants
CONFIG_FILE = "tinyfabulist.yaml"

logger = setup_logging()


def load_settings() -> dict:
    """
    Loads settings from the configuration file.
    Returns a dictionary containing all configuration settings.
    Raises ConfigError if the file is not found or has invalid YAML format.
    """
    try:
        with open(CONFIG_FILE, "r") as file:
            settings = yaml.safe_load(file)
            logger.info("Settings loaded successfully")
            return settings
    except FileNotFoundError:
        logger.error(f"Settings file '{CONFIG_FILE}' not found")
        raise ConfigError(f"Settings file '{CONFIG_FILE}' not found")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise ConfigError(f"Invalid YAML format: {e}")
