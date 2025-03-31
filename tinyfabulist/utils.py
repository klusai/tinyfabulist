import os
import glob
import yaml
from collections import defaultdict

from tinyfabulist.logger import *

# Constants
CONFIG_DIR = "tinyfabulist/conf"

logger = setup_logging()

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass

def deep_update(source, update):
    """
    Recursively update nested dictionaries
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = deep_update(source[key], value)
        else:
            source[key] = value
    return source

def load_settings() -> dict:
    """
    Loads settings from all YAML files in the conf directory.
    Returns a dictionary containing all configuration settings merged together.
    Raises ConfigError if the directory is not found or files have invalid YAML format.
    """
    settings = {}
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CONFIG_DIR)
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration directory '{config_path}' not found")
        raise ConfigError(f"Configuration directory '{config_path}' not found")
    
    config_files = glob.glob(os.path.join(config_path, "*.yaml"))
    
    if not config_files:
        logger.error(f"No YAML configuration files found in '{config_path}'")
        raise ConfigError(f"No YAML configuration files found in '{config_path}'")
    
    for config_file in sorted(config_files):
        try:
            with open(config_file, "r") as file:
                file_settings = yaml.safe_load(file)
                if file_settings:
                    settings = deep_update(settings, file_settings)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_file}: {e}")
            raise ConfigError(f"Invalid YAML format in {os.path.basename(config_file)}: {e}")
    
    logger.info(f"Successfully loaded settings from {len(config_files)} configuration files")
    return settings

def load_specific_config(config_name):
    """
    Loads a specific configuration file from the conf directory.
    Args:
        config_name: Name of the configuration file (without .yaml extension)
    Returns:
        Dictionary containing the configuration settings
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CONFIG_DIR)
    config_file = os.path.join(config_path, f"{config_name}.yaml")
    
    try:
        with open(config_file, "r") as file:
            settings = yaml.safe_load(file)
            logger.info(f"Loaded specific configuration from {config_name}.yaml")
            return settings
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_name}.yaml' not found")
        raise ConfigError(f"Configuration file '{config_name}.yaml' not found")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file}: {e}")
        raise ConfigError(f"Invalid YAML format in {config_name}.yaml: {e}")