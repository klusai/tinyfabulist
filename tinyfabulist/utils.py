import os
import glob
import yaml
from collections import defaultdict
import subprocess
import time
import datetime
from pathlib import Path

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

def get_major_version() -> str:
    """
    Read the major version from the version.yaml config file.
    Returns the major version as a string, defaulting to "0.1" if not found.
    """
    try:
        version_config = load_specific_config("version")
        return version_config.get("version", {}).get("major", "0.1")
    except ConfigError as e:
        logger.warning(f"Could not load version config: {e}. Using default major version 0.1")
        return "0.1"

import datetime


def get_version_info():
    """
    Calculate the current version of the package in the format:
    major_version.commit_count.date (e.g., 0.1.42.230512)

    Returns:
        dict: A dictionary containing version info with the following keys:
            - 'version': The full version string
            - 'major_version': The manually set major version
            - 'commit_count': The number of commits (fallback to 0)
            - 'date': The date in YYMMDD format
            - 'last_commit_hash': 'unknown'
            - 'last_commit_msg': 'unknown'
    """
    # Get major version from config or constant
    try:
        from tinyfabulist.utils import get_major_version
        major_version = get_major_version()
    except ImportError:
        major_version = "0.1"

    # Always use current date
    date_str = datetime.datetime.now().strftime("%y%m%d")

    # Fallback values since git is not used
    commit_count = "0"
    last_commit_hash = "unknown"
    last_commit_msg = "unknown"

    # Build the version string
    version = f"{major_version}.{commit_count}.{date_str}"

    return {
        'version': version,
        'major_version': major_version,
        'commit_count': commit_count,
        'date': date_str,
        'last_commit_hash': last_commit_hash,
        'last_commit_msg': last_commit_msg
    }


def update_changelog(version_info=None):
    """
    Update the CHANGELOG.md file with the latest version information.
    
    Args:
        version_info (dict, optional): Version info dictionary. If None, it will be calculated.
        
    Returns:
        str: The path to the changelog file
    """
    if version_info is None:
        version_info = get_version_info()
    
    changelog_path = Path("CHANGELOG.md")
    
    # Create changelog file if it doesn't exist
    if not changelog_path.exists():
        with open(changelog_path, "w") as f:
            f.write("# Changelog\n\n")
            f.write("All notable changes to this project will be documented in this file.\n\n")
    
    # Read existing changelog
    with open(changelog_path, "r") as f:
        content = f.read()
    
    # Check if this version is already in the changelog
    version_header = f"## [{version_info['version']}]"
    if version_header in content:
        return str(changelog_path)
    
    # Get the timestamp in ISO format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build the new entry
    new_entry = f"{version_header} - {timestamp}\n\n"
    
    # Add commit information
    if version_info['last_commit_hash'] != "unknown":
        new_entry += f"- Commit: {version_info['last_commit_hash']} - {version_info['last_commit_msg']}\n\n"
    
    # Find the position to insert the new entry (after the header and description)
    lines = content.split("\n")
    insert_pos = 0
    
    for i, line in enumerate(lines):
        if line.startswith("## ["):
            insert_pos = i
            break
        elif i > 5:  # If we've checked several lines and not found a version header
            insert_pos = i
    
    # Insert the new entry
    if insert_pos > 0:
        updated_content = "\n".join(lines[:insert_pos]) + "\n" + new_entry + "\n".join(lines[insert_pos:])
    else:
        updated_content = content + "\n" + new_entry
    
    # Write updated changelog
    with open(changelog_path, "w") as f:
        f.write(updated_content)
    
    return str(changelog_path)
